import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
st.set_page_config(
    page_title="FitPulse",
    layout="wide",
    page_icon="⚡",
    initial_sidebar_state="expanded"
)

BACKEND_URL = "http://127.0.0.1:8000"

# ------------------------------------------------------------------
# PROFESSIONAL ENTERPRISE CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    background-color: #010028;
    color: white;
}

.stApp {
    background-color: #010028;
}

/* ---------------- SIDEBAR ---------------- */
[data-testid="stSidebar"] {
    background-color: rgba(1, 0, 40, 1);
    padding-top: 1.5rem;
    border-right: 1px solid rgba(255, 255, 255, 0.5);
}
[data-testid="stSidebar"] * {
    color: white;
}
[data-testid="stSidebar"] span {
    color: rgba(1, 1, 74, 1);
}
[data-baseweb="textarea"] {
    color: #01014a;
}
[data-testid="stSidebar"] button {
    background-color: #62eec7;
    color: #01014a;
    border: 1px solid #01014a
}
[data-testid="stSidebarCollapseButton"] {
    color: #62eec7;
}
[data-testid="stIconMaterial"] {
    color: #62eec7;
    z-index: 99999;
}
[data-test-id="stChatMessage"] {
    color: #01014a;
    background-color: #62eec7;
}
[data-testid="stFileUploaderDropzone"] {
    background-color: #FFFFFF;
    color: #01014a;
}
[data-test-id="stLayoutWrapper"] {
    color: #01014a;
    background-color: #62eec7;
}

[data-testid="stChatInputTextArea"] {
    color: #01014a;
}

[data-testid="stChatMessageContent"] code {
    color: #62eec7;
    background-color: #010028;
}
                          
/* ---------------- HEADER / HERO ---------------- */
.hero {
    background-color: white;
    padding: 28px 36px;
    border-radius: 12px;
    border-left: 6px solid #62eec7;
    box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    margin-bottom: 30px;
    color: #01014a;
}
.hero h1{
    color: #01014a;
    font-family: Garamond bold;
}

.info p {
    font-size: 20px;
    padding: 14px;
    color: rgba(255,100,0,0.8);
}

/* ---------------- CARDS ---------------- */
.card {
    background-color: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    border: 1px solid #FFFFFF;
}

/* ---------------- KPI ---------------- */
.metric-value {
    font-size: 36px;
    font-weight: 700;
    color: #1E3A8A;
}
.metric-label {
    font-size: 13px;
    color: #6B7280;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ---------------- BUTTONS ---------------- */
.stButton > button {
    background-color: #62eec7;
    color: #01014a;
    border-radius: 10px;
    font-weight: 600;
    border: 1px solid #01014a;
    margin-top: 2px;
    height: 43px;
}
            
.stButton > button:hover {
    background-color: #D59ADE;
}

[data-testid = "stBaseButton-secondary"] {
    background-color: #62eec7;
    color: #01014a;
    border: 1px solid #01014a;
}
            
[data-testid = "stBaseButton-secondary"]:hover {
    background-color: #D59ADE;
}

            
/* ---------------- TABS ---------------- */
.stTabs [data-baseweb="tab"] {
    background-color: #62eec7;
    border-radius: 6px;
    padding: 10px 18px;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background-color: white !important;
    border-bottom: 3px solid #1E3A8A;
}
     

/* Rounded Plotly charts */
div[data-testid="stPlotlyChart"] {
    border-radius: 14px;
    overflow: hidden;
    border: 2px solid #62eec7;
    background-color: white;
}

/* ---------------- TABLES ---------------- */
.stDataFrame {
    background-color: white;
    border-radius: 10px;
    border: 1px solid #FFFFFF;
}

/* ---------------- STATUS ---------------- */
[data-testid="stStatusWidget"] {
    background-color: white;
    border: 1px solid #E5E7EB;
    color: #1F2937;
}

[data-testid="stFileUploaderFile"] div {
    color: rgba(255, 255, 255, 0.8);
}

[data-testid="stFileUploaderFile"] div small {
    color: rgba(255, 255, 255, 0.5);
}

[data-testid="stFileUploader"] {
    margin-bottom: 20px;
}
            
.st-key-upload p, .st-key-option p {
    color: #62eec7; 
    font-size: 16px;
    font-weight: 500;     
}

[data-testid = "stHeadingWithActionElements"] span {
    color: white;
}

            
[data-testid = "stWidgetLabel"] p {
    color: white;
}
            
[data-testid = "stAlertContentInfo"] p {
    color: rgba(255, 100, 0, 0.7);
}
  
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# SESSION STATE
# ------------------------------------------------------------------
if "data" not in st.session_state:
    st.session_state.data = None
if "forecast_ready" not in st.session_state:
    st.session_state.forecast_ready = False
if "cluster_ready" not in st.session_state:
    st.session_state.cluster_ready = False


# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
with st.sidebar:
    st.title("FitPulse Analytical Assisstant")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=False)

    # Accept user input
    if prompt := st.chat_input("Ask about your fitness data..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        try:
            # Call backend AI endpoint
            current_user = st.session_state.get("user_filter", "All Users")
            response = requests.post(
                f"{BACKEND_URL}/ask-ai",
                params={"user_message": prompt, "user_id": str(current_user)}
            )
            
            if response.status_code == 200:
                ai_response = response.json()["response"] 
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
            else:
                error_msg = "⚠️ Please upload and process data first before asking questions."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        
        # Rerun to display messages in correct order
        st.rerun()


# ------------------------------------------------------------------
# HERO HEADER
# ------------------------------------------------------------------
st.markdown("""
    <div style="text-align:left;">
        <span><h2 style="font-size:45px;font-weight:600;font-family:Garamond;color:white;">⚡FitPulse</h2></span>
        <span><p style="opacity:0.8;font-size:18px;font-style:italic;font-family:Times New Roman;color:white;margin-left:40px;margin-bottom:30px;margin-top:-10px;">--- " Fitness for All "</p></span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="hero" style="color: #01014a;">
    <p style="font-size: 50px; font-weight: 750; font-family: Garamond;">FitPulse Analytics Dashboard</p>
    <p>
        Enterprise-grade health analytics platform for monitoring,
        forecasting, and anomaly detection.
    </p>
</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------
# DASHBOARD
# ------------------------------------------------------------------
if st.session_state.data is not None:
    # --- USER FILTER ---
    all_users = sorted(st.session_state.data['user_id'].unique())
    user_options = ["All Users"] + list(all_users)
    
    col_filter, col_rest = st.columns([1, 4])
    with col_filter:
        st.markdown("### 🔍 Filter")
        selected_user = st.selectbox("Select User ID", user_options, key="user_filter")
        
    if selected_user != "All Users":
        df = st.session_state.data[st.session_state.data['user_id'] == selected_user].copy()
    else:
        df = st.session_state.data.copy()

    users = df["user_id"].nunique()
    records = len(df)
    avg_hr = int(df["heart_rate"].mean())
    avg_steps = int(df["steps"].mean())

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in zip(
        [c1, c2, c3, c4],
        [users, records, avg_hr, avg_steps],
        ["Active Users", "Total Records", "Avg Heart Rate", "Avg Steps"]
    ):
        with col:
            st.markdown(f"""
            <div class="card" style="text-align:center;">
                <div class="metric-value">{val:,}</div>
                <div class="metric-label">{label}</div>
            </div>
            <br>
            """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["Data Overview", "Forecasting", "User Segmentation","Health Insights"]
    )

    with tab1:
        st.dataframe(df, use_container_width=True)

    with tab2:
        if st.session_state.forecast_ready:
            st.markdown("##### Prophet Forecasting (Seasonality + Anomalies)")
            
            # --- FETCH FORECAST (Global or Specific) ---
            forecast_source = st.session_state.forecast_data
            
            if selected_user != "All Users":
                try:
                    params = {"user_id": str(selected_user)} # Pass as string explicitly
                    with st.spinner(f"Generating forecast for User {selected_user}..."):
                        f_res = requests.post(f"{BACKEND_URL}/prophet-forecast", params=params)
                        if f_res.status_code == 200:
                            forecast_source = f_res.json().get("metrics")
                        elif f_res.status_code == 400:
                             st.error("⚠️ Backend data lost. Please click 'Run Analysis Pipeline' in the sidebar to re-process.")
                        else:
                            st.warning("Could not fetch user specific forecast.")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            
            metrics = forecast_source
            
            if metrics:
                # Removed columns to stack vertically for better visibility
                
                # Helper for Plotly Charts
                def render_forecast_chart(metric_name, title, color, unit):
                    data = metrics.get(metric_name)
                    if not data: 
                        st.info(f"Not enough data to forecast {metric_name} for this user.")
                        return
                    
                    f_df = pd.DataFrame(data)
                    f_df['ds'] = pd.to_datetime(f_df['ds'])
                    
                    st.markdown(f"### {title}") # Explicit Heading
                    
                    fig = go.Figure()
                    
                    # 1. Uncertainty Interval (Cloud)
                    fig.add_trace(go.Scatter(
                        x=pd.concat([f_df['ds'], f_df['ds'][::-1]]),
                        y=pd.concat([f_df['yhat_upper'], f_df['yhat_lower'][::-1]]),
                        fill='toself',
                        fillcolor=f"rgba({int(color[1:3],16)}, {int(color[3:5],16)}, {int(color[5:7],16)}, 0.1)",
                        line=dict(color='rgba(255,255,255,0)'),
                        hoverinfo="skip",
                        showlegend=False
                    ))
                    
                    # 2. Main Forecast Line
                    fig.add_trace(go.Scatter(
                        x=f_df['ds'], y=f_df['yhat'], 
                        mode='lines', 
                        name='Forecast', 
                        line=dict(color=color, width=3)
                    ))
                    
                    # 3. Actual Data (Dots)
                    fig.add_trace(go.Scatter(
                        x=f_df['ds'], y=f_df['y'],
                        mode='markers',
                        name='Actual',
                        marker=dict(color='black', size=4, opacity=0.5)
                    ))
                    
                    # 4. Anomalies (Big Red Dots)
                    anoms = f_df[f_df['anomaly'] == True]
                    fig.add_trace(go.Scatter(
                        x=anoms['ds'], y=anoms['y'], 
                        mode='markers', 
                        name='Anomaly', 
                        marker=dict(color='red', size=10, symbol='x')
                    ))
                    
                    fig.update_layout(
                        height=400, 
                        margin=dict(l=20, r=20, t=10, b=20), 
                        hovermode="x unified",
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1f2937'),
                        xaxis=dict(showgrid=False, gridcolor='#eee', tickfont=dict(color='#1f2937')),
                        yaxis=dict(showgrid=True, gridcolor='#eee', title=unit, tickfont=dict(color='#1f2937'))
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Helper for Weekly Seasonality (Steps)
                def render_weekly_chart(metric_name, title, color):
                    data = metrics.get(metric_name)
                    if not data: return
                    
                    df = pd.DataFrame(data)
                    if "weekly" not in df.columns:
                        st.info(f"Not enough data for weekly seasonality in {metric_name}")
                        return

                    # Create a daily pattern (Monday=0 ... Sunday=6)
                    # We just take the unique weekly values associated with each weekday 
                    # from the forecast to represent the "component"
                    df['ds'] = pd.to_datetime(df['ds'])
                    df['day_name'] = df['ds'].dt.day_name()
                    df['day_index'] = df['ds'].dt.dayofweek
                    
                    # Aggregate average effect per day
                    weekly_df = df.groupby(['day_name', 'day_index'])['weekly'].mean().reset_index()
                    weekly_df = weekly_df.sort_values('day_index')
                    
                    st.markdown(f"### {title}")
                    fig = px.bar(
                        weekly_df, x='day_name', y='weekly', 
                        title="Day of Week Impact",
                        labels={'weekly': 'Effect on Steps', 'day_name': 'Day'},
                        color_discrete_sequence=[color]
                    )
                    fig.update_layout(
                        height=400,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1f2937'),
                        xaxis=dict(tickfont=dict(color='#1f2937')),
                        yaxis=dict(tickfont=dict(color='#1f2937'))
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Helper for Trend Component (Sleep)
                def render_trend_chart(metric_name, title, color):
                    data = metrics.get(metric_name)
                    if not data: return
                    
                    df = pd.DataFrame(data)
                    df['ds'] = pd.to_datetime(df['ds'])
                    
                    st.markdown(f"### {title}")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=df['ds'], y=df['trend'], mode='lines', name='Trend', line=dict(color=color, width=4)))
                    
                    fig.update_layout(
                        title="Long-Term Trend",
                        height=400,
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(color='#1f2937'),
                        xaxis=dict(showgrid=False, gridcolor='#eee', tickfont=dict(color='#1f2937')),
                        yaxis=dict(showgrid=True, gridcolor='#eee', tickfont=dict(color='#1f2937'))
                    )
                    st.plotly_chart(fig, use_container_width=True)


                # Render Separately with Spacers
                render_forecast_chart("heart_rate", "❤️ Heart Rate Forecast", "#ef4444", "BPM")
                st.markdown("---")
                
                render_weekly_chart("steps", "👟 Step Count - Weekly Pattern", "#3b82f6")
                st.markdown("---")
                
                render_trend_chart("sleep", "🌙 Sleep Duration - Long Term Trend", "#8b5cf6")
                
            else:
                st.info("No forecast data available.")
                
        else:
            st.warning("Forecast data not generated yet. Run the pipeline.")

    with tab3:
        if st.session_state.cluster_ready:
            cdf = st.session_state.cluster_data
            fig = px.scatter(
                cdf,
                x="steps",
                y="heart_rate",
                color="cluster",
                hover_data=["user_id"]
            )
            fig.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=420,
                font=dict(color="#1F2937")
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab4:
        if selected_user == "All Users":
            st.info("ℹ️ Please select a specific User ID to generate a personalized health report.")
        else:
            st.markdown(f"### 🩺 Health Analysis: User {selected_user}")
            
            try:
                params = {"user_id": str(selected_user)}
                i_res = requests.post(f"{BACKEND_URL}/user-insights", params=params)
                
                if i_res.status_code == 200:
                    report = i_res.json()
                    
                    if "error" in report:
                        st.error(report["error"])
                    else:
                        # Display Label
                        status = report.get("label", "Balanced")
                        st.markdown(f"""
                        <div class="card" style="border-left: 5px solid #2563eb;">
                            <p style="color: #01014a; font-size: 20px;">Health Status Profile</p>
                            <p style="color: #01014a; font-size: 40px;">{status}</p>
                        </div>
                        <br>
                        """, unsafe_allow_html=True) 
                        
                        col_a, col_b = st.columns(2, gap="xsmall")
                        
                        with col_a:
                            st.subheader("🧐 :orange[Observations]")
                            for obs in report.get("observations", []):
                                st.write(f":orange[- {obs}]")
                                
                        with col_b:
                            st.subheader("🤖 :green[AI Suggestions]")
                            for sug in report.get("suggestions", []):
                                st.success(f"💡 :green[{sug}]")
                                
                elif i_res.status_code == 400:
                    st.error("⚠️ Backend data lost. Please click 'Run Analysis Pipeline' again.")
                else:
                    st.error("Failed to fetch insights.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="upload")
    col1, col2 = st.columns(2)
    with col1:
        users = ["All"] + [i for i in range(0,500)]
        option = st.selectbox(
            "Which users do you want to select?",
            tuple(users),
            key="option",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        start_btn = st.button("Run Analysis Pipeline", key="b1")
    st.markdown("""     
        <div class="info" style="margin-top:-18px;margin-left:-15px;">
            <p>ℹ️ Upload a CSV file to start analysis.</p>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file and start_btn:
        uploaded_file.seek(0)
        files = {"file": uploaded_file}

        with st.status("Executing Data Pipeline...", expanded=True):
            requests.post(f"{BACKEND_URL}/preprocess", files=files)

            bar = st.progress(0)
            while True:
                p = requests.get(f"{BACKEND_URL}/progress").json()
                bar.progress(p["progress"] / 100)
                if p["progress"] == 100:
                    st.session_state.data = pd.DataFrame(p["data"])
                    break
                time.sleep(0.4)

            requests.post(f"{BACKEND_URL}/feature-extraction")
            f = requests.post(f"{BACKEND_URL}/prophet-forecast")
            c = requests.post(f"{BACKEND_URL}/cluster-anomaly")

            if f.status_code == 200:
                st.session_state.forecast_data = f.json()["metrics"]
                st.session_state.forecast_ready = True
            if c.status_code == 200:
                st.session_state.cluster_data = pd.DataFrame(c.json()["summary"])
                st.session_state.cluster_ready = True
        
        st.rerun()
