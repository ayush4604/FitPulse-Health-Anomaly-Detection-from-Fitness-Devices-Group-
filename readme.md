# ⚡ FitPulse – AI-Powered Fitness Analytics Platform

FitPulse is an **enterprise-grade fitness analytics web application** that transforms raw wearable health data into **interactive dashboards, forecasts, anomaly detection insights, and AI-generated fitness explanations**.

It combines **time-series analytics, machine learning, and LLM-based natural language insights** using a **FastAPI backend** and a **Streamlit frontend**.

> ⚠️ This project is designed for **fitness analytics and educational purposes only**.  
> It does **not** provide medical diagnosis or treatment recommendations.

---

## 🚀 Key Features

- 📊 **Interactive Fitness Dashboard**
  - Heart rate, steps, and sleep analytics
  - User-level filtering and KPIs

- 📈 **Time-Series Forecasting**
  - Facebook Prophet for trend, seasonality, and anomaly detection
  - 30-day future forecasts

- 🧠 **Machine Learning Analytics**
  - TSFresh feature extraction
  - DBSCAN-based user clustering and anomaly detection

- 🤖 **AI Fitness Assistant**
  - Natural language explanations powered by **Groq LLM**
  - Context-aware insights based strictly on dataset analytics

- 📄 **Automated PDF Fitness Reports**
  - Rule-based insights + AI narrative
  - Downloadable professional reports per user

---

## 🗂️ Project Structure
├── data/
│ └── health_fitness_tracking_365days.csv
│
├── screenshots/
│ └── (Screenshots of the web application UI)
│
├── final_backend.py
├── final_frontend.py
├── requirements.txt
├── .env # Not included in GitHub (required locally)
├── README.md
└── LICENSE # MIT License


---

## 🧱 System Architecture

User → Streamlit Frontend → FastAPI Backend → ML Models (Prophet, DBSCAN, TSFresh) → Groq LLM (AI Insights) → PDF Report Generator


---

## 📊 Dataset

- **File**: `health_fitness_tracking_365days.csv`
- **Duration**: 365 days
- **Key Columns**:
  - `user_id`
  - `timestamp / date`
  - `heart_rate`
  - `steps`
  - `sleep`

The backend automatically standardizes column names during preprocessing.

---

## ⚙️ Tech Stack

### Frontend
- Streamlit
- Plotly (interactive charts)
- Custom enterprise-grade CSS

### Backend
- FastAPI
- Background tasks & async processing

### Machine Learning
- Facebook Prophet (forecasting & anomalies)
- TSFresh (feature extraction)
- DBSCAN (clustering & anomaly detection)
- Scikit-learn

### AI / LLM
- Groq API
- LangChain

### Reporting
- ReportLab (PDF generation)

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
pip install -r requirements.txt
```

Create a .env file locally:
```bash
GROQ_API_KEY=your_groq_api_key_here
```

##▶️ Running the Application

###1️⃣ Start Backend (FastAPI)
```bash
uvicorn final_backend:app --reload
```
Backend runs at: http://127.0.0.1:8000

###2️⃣ Start Frontend (Streamlit)
```bash
streamlit run final_frontend.py
```
Frontend runs at: http://localhost:8501

---

##🧪 How to Use

1.  Upload the fitness CSV file
2. Click Run Analysis Pipeline
3. Explore:
  - Data overview
  - Forecasting & anomalies
  - User clustering
  - Personalized health insights
4. Ask questions using the AI Fitness Assistant
5. Download AI-generated PDF fitness reports

---

##🧠 AI Safety & Constraints

The AI assistant:
- Uses only provided analytics
- Does not perform medical diagnosis
- Provides non-medical fitness insights
- Clearly states uncertainty when data is limited

📸 Screenshots

Screenshots of the dashboard, forecasting charts, clustering plots, and AI insights are present in the screenshots folder.

📄 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute this project with attribution.

👨‍💻 Authors
Pranavakumar Murali, Ayush Kumar, Nagararaj, Neha Joshi, Hima Priya

FitPulse
An end-to-end AI + ML fitness analytics system built using
FastAPI, Streamlit, Machine Learning, and LLMs
