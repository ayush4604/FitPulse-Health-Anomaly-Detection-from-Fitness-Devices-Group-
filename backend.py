
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
import pandas as pd
import numpy as np
import traceback
from prophet import Prophet
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

app = FastAPI(title="FitPulse – Fast Backend")

# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CLEAN_DF = None
FEATURE_DF = None


# -------------------------------
# ERROR HANDLER
# -------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    trace = traceback.format_exc()
    print(trace)
    return JSONResponse(
        status_code=500,
        content={"error": "Backend crashed", "trace": trace}
    )


# -------------------------------
# PREPROCESS (FAST)
# -------------------------------
@app.post("/preprocess")
async def preprocess(file: UploadFile = File(...)):
    global CLEAN_DF

    df = pd.read_csv(file.file)

    df = df.rename(columns={
        "date_time": "timestamp",
        "date": "timestamp",
        "heart_rate_avg": "heart_rate",
        "step_count": "steps",
        "total_steps": "steps",
        "sleep_hours": "sleep"
    })

    REQUIRED = ["timestamp", "user_id", "heart_rate", "steps", "sleep"]
    for c in REQUIRED:
        if c not in df.columns:
            return JSONResponse(status_code=400, content={"error": f"Missing {c}"})

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce").fillna(df["heart_rate"].median())
    df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
    df["sleep"] = pd.to_numeric(df["sleep"], errors="coerce").fillna(df["sleep"].median())

    # Resample to daily average for consistency
    df = (
        df.set_index("timestamp")
        .groupby("user_id")[["heart_rate", "steps", "sleep"]]
        .resample("D")
        .mean()
        .reset_index()
    )

    CLEAN_DF = df
    return {"status": "success", "rows": len(df)}


# -------------------------------
# FEATURE EXTRACTION (TSFresh)
# -------------------------------
def _feature_extraction_job():
    global CLEAN_DF, FEATURE_DF

    df = CLEAN_DF.copy()

    # Fill NaNs before TSFresh
    df = df.fillna(0)

    # Use MinimalFCParameters for efficiency
    settings = MinimalFCParameters()

    # Re-structure for TSFresh (long format)
    df_long = df.melt(id_vars=["user_id", "timestamp"], value_vars=["heart_rate", "steps", "sleep"],
                      var_name="kind", value_name="value")

    extracted_features = extract_features(
        timeseries_container=df_long,
        column_id="user_id",
        column_sort="timestamp",
        column_kind="kind",
        column_value="value",
        default_fc_parameters=settings
    )

    # FEATURE_DF for Prophet needs to be the time series data
    df_ts = df.sort_values(["user_id", "timestamp"]).copy()

    # Add manual features to time series for per-point analysis
    df_ts["hr_7d_mean"] = df_ts.groupby("user_id")["heart_rate"].transform(lambda x: x.rolling(7, min_periods=1).mean())
    df_ts["hr_7d_std"] = df_ts.groupby("user_id")["heart_rate"].transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0))

    FEATURE_DF = df_ts

    return len(extracted_features)


@app.post("/feature-extraction")
async def feature_extraction():
    if CLEAN_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /preprocess first"})

    num_features = await run_in_threadpool(_feature_extraction_job)

    return {"status": "success", "rows": len(FEATURE_DF), "tsfresh_users_processed": num_features}


# -------------------------------
# PROPHET (Seasonality + Anomalies)
# -------------------------------
def _prophet_job():
    global FEATURE_DF

    results = {}
    METRICS = ["heart_rate", "steps", "sleep"]

    for metric in METRICS:
        p_df = (
            FEATURE_DF.groupby("timestamp")[metric]
            .mean()
            .reset_index()
            .rename(columns={"timestamp": "ds", metric: "y"})
        )

        if len(p_df) <= 10:
            continue

        model = Prophet()
        model.fit(p_df)

        # Forecast on history for anomalies
        forecast = model.predict(p_df)
        p_df["yhat"] = forecast["yhat"]
        p_df["yhat_lower"] = forecast["yhat_lower"]
        p_df["yhat_upper"] = forecast["yhat_upper"]

        p_df["anomaly"] = (p_df["y"] < p_df["yhat_lower"]) | (p_df["y"] > p_df["yhat_upper"])

        # Future forecast
        future = model.make_future_dataframe(periods=30, freq='D')
        future_forecast = model.predict(future)

        # Combine
        combined = pd.concat([p_df, future_forecast.iloc[len(p_df):]])

        # Select relevant columns
        combined = combined[["ds", "y", "yhat", "yhat_lower", "yhat_upper", "anomaly"]]

        combined["anomaly"] = combined["anomaly"].fillna(False)
        combined["y"] = combined["y"].fillna(0)
        combined = combined.fillna(0)

        results[metric] = combined.tail(60).to_dict("records")

    return results


@app.post("/prophet-forecast")
async def prophet_forecast():
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /feature-extraction first"})

    results = await run_in_threadpool(_prophet_job)

    return {"status": "success", "metrics": results}


# -------------------------------
# Clustering + Anomaly-Detection
# -------------------------------

def _cluster_anomaly_job():

    def dbscan_clustering(df):

        user_features = (
            df.groupby("user_id")[["heart_rate", "steps", "sleep"]]
            .mean()
        )

        scaler = StandardScaler()
        X = scaler.fit_transform(user_features)

        dbscan = DBSCAN(eps=1.2, min_samples=10)
        clusters = dbscan.fit_predict(X)

        user_features["cluster"] = clusters

        return user_features.reset_index()


    def detect_cluster_anomalies(cluster_df):
        """
        DBSCAN-based anomaly detection.
        Noise points (-1) are anomalies.
        """

        cluster_df["cluster_anomaly"] = cluster_df["cluster"] == -1
        return cluster_df

    # Step 1: clustering
    cluster_df = dbscan_clustering(FEATURE_DF)

    # Step 2: anomaly detection
    cluster_df = detect_cluster_anomalies(cluster_df)

    # Step 3: map back
    FEATURE_DF["cluster"] = FEATURE_DF["user_id"].map(
        cluster_df.set_index("user_id")["cluster"]
    )

    FEATURE_DF["cluster_anomaly"] = FEATURE_DF["user_id"].map(
        cluster_df.set_index("user_id")["cluster_anomaly"])


@app.post("/cluster-anomaly")
async def cluster_anomaly():
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /feature-extraction first"})

    results = await run_in_threadpool(_cluster_anomaly_job)

    return {
        "status": "success",
        "summary": results
    }
