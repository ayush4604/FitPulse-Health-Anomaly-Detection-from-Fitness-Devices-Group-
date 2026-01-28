
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi import BackgroundTasks
import pandas as pd
import numpy as np
import traceback
import time
from prophet import Prophet
from tsfresh import extract_features
from tsfresh.feature_extraction import MinimalFCParameters
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(
            model = "openai/gpt-oss-20b",
            groq_api_key=groq_api_key,
        )

app = FastAPI(title="FitPulse – Fast Backend")
JOB_STATUS = {"progress": 0, "status": "idle", "data": None}

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
# -------------------------------
# PREPROCESS (Background Task)
# -------------------------------
def preprocessing_task(df: pd.DataFrame):
    global CLEAN_DF, JOB_STATUS
    
    try:
        JOB_STATUS = {"progress": 10, "status": "Cleaning Data..."}

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
                JOB_STATUS = {"progress": 0, "status": f"Error: Missing {c}"}
                return

        JOB_STATUS["progress"] = 30

        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])

        df["heart_rate"] = pd.to_numeric(df["heart_rate"], errors="coerce").fillna(df["heart_rate"].median())
        df["steps"] = pd.to_numeric(df["steps"], errors="coerce").fillna(0)
        df["sleep"] = pd.to_numeric(df["sleep"], errors="coerce").fillna(df["sleep"].median())

        JOB_STATUS["progress"] = 60

        # Resample to daily average for consistency
        df = (
            df.set_index("timestamp")
            .groupby("user_id")[["heart_rate", "steps", "sleep"]]
            .resample("D")
            .mean()
            .reset_index()
        )

        CLEAN_DF = df
        JOB_STATUS = {
            "progress": 100, 
            "status": "Completed", 
            "rows": len(df), 
            "data": df.to_dict(orient="records")
        }
        
    except Exception as e:
        traceback.print_exc()
        JOB_STATUS = {"progress": 0, "status": f"Error: {str(e)}"}


@app.post("/preprocess")
async def preprocess(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    global JOB_STATUS
    
    # Reset Status
    JOB_STATUS = {"progress": 1, "status": "Starting..."}
    
    # Read file synchronously before passing to background task
    df = pd.read_csv(file.file)
    
    background_tasks.add_task(preprocessing_task, df)
    
    return {"message": "Preprocessing started"}


@app.get("/progress")
def get_progress():
    return JOB_STATUS


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
def _prophet_job(target_user=None):
    global FEATURE_DF
    
    # Work on a copy
    df_work = FEATURE_DF.copy()
    
    # Filter by user if specified
    if target_user is not None:
        # Try numeric conversion if applicable
        try:
            u_id = int(target_user)
            df_work = df_work[df_work["user_id"] == u_id]
        except:
            df_work = df_work[df_work["user_id"] == target_user]
            
        if df_work.empty:
            return {}

    results = {}
    METRICS = ["heart_rate", "steps", "sleep"]

    for metric in METRICS:
        p_df = (
            df_work.groupby("timestamp")[metric]
            .mean()
            .reset_index()
            .rename(columns={"timestamp": "ds", metric: "y"})
        )

        if len(p_df) <= 5: # Lower threshold for single user
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
        cols_to_keep = ["ds", "y", "yhat", "yhat_lower", "yhat_upper", "anomaly", "trend"]
        if "weekly" in combined.columns:
            cols_to_keep.append("weekly")
            
        combined = combined[cols_to_keep]

        combined["anomaly"] = combined["anomaly"].fillna(False)
        combined["y"] = combined["y"].fillna(0)
        combined = combined.fillna(0)

        results[metric] = combined.tail(60).to_dict("records")

    return results


@app.post("/prophet-forecast")
async def prophet_forecast(user_id: str = None):
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /feature-extraction first"})

    results = await run_in_threadpool(_prophet_job, user_id)

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

    return cluster_df.to_dict(orient="records")


@app.post("/cluster-anomaly")
async def cluster_anomaly():
    if FEATURE_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /feature-extraction first"})

    results = await run_in_threadpool(_cluster_anomaly_job)

    return {
        "status": "success",
        "summary": results
    }


# -------------------------------
# HEALTH INSIGHTS ENGINE
# -------------------------------
def _generate_insights_job(target_user):
    global CLEAN_DF
    
    # 1. Access Data
    df = CLEAN_DF.copy()
    
    # 2. Filter User
    try:
        u_id = int(target_user)
        user_df = df[df["user_id"] == u_id]
    except:
        user_df = df[df["user_id"] == target_user]
        
    if user_df.empty:
        return {"error": "User not found"}
        
    # 3. Calc Stats
    avg_hr = user_df["heart_rate"].mean()
    avg_steps = user_df["steps"].mean()
    avg_sleep = user_df["sleep"].mean()
    
    # 4. Generate Rules
    observations = []
    suggestions = []
    status_label = "Balanced"
    
    # Heart Rate Logic
    if avg_hr < 60:
        observations.append(f"Low Resting HR ({int(avg_hr)} bpm). Usually indicates good fitness.")
    elif avg_hr > 90:
        observations.append(f"Elevated HR ({int(avg_hr)} bpm).")
        suggestions.append("Consider stress management or more cardio to lower resting HR.")
    else:
        observations.append(f"Normal Heart Rate ({int(avg_hr)} bpm).")

    # Steps Logic
    if avg_steps < 5000:
        observations.append(f"Low Activity ({int(avg_steps)} steps/day).")
        suggestions.append("Aim for at least 8,000 steps daily. Try short walks.")
        status_label = "Sedentary"
    elif avg_steps > 10000:
        observations.append(f"High Activity ({int(avg_steps)} steps/day).")
        suggestions.append("Excellent activity level! Maintain this consistency.")
        status_label = "Active"
    else:
        observations.append(f"Moderate Activity ({int(avg_steps)} steps/day).")
        suggestions.append("You are close to the 10k steps goal. Push a little more!")

    # Sleep Logic
    if avg_sleep < 6:
        observations.append(f"Sleep Deprived ({avg_sleep:.1f} hrs).")
        suggestions.append("Target 7-8 hours. Avoid screens before bed.")
    elif avg_sleep > 9:
        observations.append(f"Oversleeping ({avg_sleep:.1f} hrs).")
    else:
        observations.append(f"Good Sleep Hygiene ({avg_sleep:.1f} hrs).")

    return {
        "avg_hr": avg_hr,
        "avg_steps": avg_steps,
        "avg_sleep": avg_sleep,
        "label": status_label,
        "observations": observations,
        "suggestions": suggestions
    }


@app.post("/user-insights")
async def user_insights(user_id: str):
    if CLEAN_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /preprocess first"})
        
    analysis = await run_in_threadpool(_generate_insights_job, user_id)
    return analysis


def _chat_job(user_message, user_id):
    global CLEAN_DF, FEATURE_DF

    # 1. Prepare Data Context
    df = CLEAN_DF.copy()
    context_str = "Global Dataset"
    
    # Filter by user if specific
    if user_id and str(user_id) != "All Users":
        try:
            u_id = int(user_id)
            df = df[df["user_id"] == u_id]
            context_str = f"User {user_id}"
        except:
            pass

    if df.empty:
        stats_summary = "No data available for this selection."
    else:
        # Calculate stats
        t_start = df["timestamp"].min().strftime("%Y-%m-%d")
        t_end = df["timestamp"].max().strftime("%Y-%m-%d")
        avg_hr = df["heart_rate"].mean()
        avg_steps = df["steps"].mean()
        avg_sleep = df["sleep"].mean()
        
        stats_summary = f"""
        - Context: {context_str}
        - Date Range: {t_start} to {t_end}
        - Average Heart Rate: {avg_hr:.1f} bpm
        - Average Daily Steps: {int(avg_steps)}
        - Average Sleep: {avg_sleep:.1f} hours/night
        """

    def setup_llm_chain(mes, stats):
        global llm
        prompt = ChatPromptTemplate(
            [
                ("system", """
                    You are FitPulse AI, a fitness analytics assistant designed for
                    fitness coaches and data scientists.

                    You specialize in interpreting:
                    - Time-series fitness data (365-day ranges)
                    - Heart rate, activity, and recovery metrics
                    - Anomaly detection outputs
                    - Forecasting results
                    - User clustering and segmentation

                    Your responsibilities:
                    - Explain analytical results clearly and accurately
                    - Highlight meaningful trends, seasonality, and anomalies
                    - Compare users, clusters, or time periods when relevant
                    - Provide light, non-medical fitness recommendations
                    (e.g., rest, recovery, training load adjustment, hydration)

                    Strict constraints:
                    - Do NOT perform medical diagnosis
                    - Do NOT prescribe medication or treatment
                    - Do NOT infer data that is not explicitly provided
                    - Do NOT generalize beyond the given analytics
                    - Always state uncertainty when data is limited

                    Tone & style:
                    - Professional and analytical
                    - Concise but insightful
                    - Coach- and data-scientist-friendly
                    - Use bullet points, tables, or structured reasoning when helpful

                    Safety:
                    - Frame all insights as fitness observations
                    - For persistent or extreme anomalies, suggest monitoring
                    or consulting qualified professionals without alarmist language

                    If a question cannot be answered using the provided data,
                    explicitly request the missing information.

                """),
                ("user", f"""
                    USER QUESTION:                
                    {mes}

                    DATA CONTEXT:
                    {stats}

                    INSTRUCTIONS:
                    Respond to the user question using ONLY the provided analytics.
                    Explain key insights, anomalies, and relevant patterns.
                    Provide light fitness recommendations when appropriate.
                """),
            ]
        )

        return prompt | llm | StrOutputParser()

    response = setup_llm_chain(user_message, stats_summary).invoke({}).strip()

    return response

@app.post("/ask-ai")
async def ask_ai(user_message: str, user_id: str = "All Users"):
    if CLEAN_DF is None:
        return JSONResponse(status_code=400, content={"error": "Run /preprocess first"})
        
    chat = await run_in_threadpool(_chat_job, user_message, user_id)
    print(chat)
    
    return {"response": chat}

