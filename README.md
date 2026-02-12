# 🏏 IPL Win Predictor

## Overview
The **IPL Win Predictor** is a machine learning-powered application designed to forecast the outcomes of Indian Premier League (IPL) cricket matches with high accuracy. By leveraging historical match data, advanced feature engineering (including ELO ratings and venue statistics), and real-time match context, this tool provides explainable win probabilities for both pre-match and live scenarios.

## 🚀 Key Features

### 1. Interactive Dashboard (Streamlit)
A user-friendly web interface built with **Streamlit** that allows users to:
- **Select Teams**: Choose any two active IPL teams for a face-off.
- **View Predictions**: Get instant win probabilities for both teams.
- **Analyze Confidence**: Visual indicators for prediction confidence (High, Medium, Low).
- **Understand the Why**: Insights into the factors driving the prediction (e.g., recent form, head-to-head ELO).

### 2. Prediction API (FastAPI)
A robust **FastAPI** backend that exposes a RESTful endpoint (`/predict`) for developers. This allows for easy integration of the prediction logic into other applications or services.
- **Endpoint**: `POST /predict`
- **Input**: JSON object containing match features.
- **Output**: JSON response with win probabilities for both teams.

### 3. Advanced Machine Learning
 The system uses dual models trained on comprehensive usage data:
- **Pre-match Model**: Prediction based on pre-game stats like team ELO, recent win rates (last 5 matches), and venue chasing history.
- **Live Model**: Real-time prediction capabilities (in `inference/`) adaptable to changing match conditions.

### 4. Sophisticated Feature Engineering
The model doesn't just look at wins and losses. It considers granular details:
- **Team ELO Ratings**: Dynamic rating system updating after every match.
- **Venue Stats**: Calculates "chasing bias" for every stadium to adjust for pitch conditions.
- **Phase-wise Performance**: Analyzes team strength in Powerplay, Middle Overs, and Death Overs.
- **Recent Form**: Sliding window analysis of a team's last 5 matches.

## 🛠️ Tech Stack
- **Language**: Python 3.x
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **ML Framework**: Scikit-learn, Joblib
- **Data Processing**: Pandas, NumPy
- **File Formats**: Parquet (Efficient data storage)

## 📦 Project Structure
```
ipl_win_predictor/
├── api/                # FastAPI backend code
│   └── main.py
├── app.py              # Streamlit frontend application
├── data/               # Processed data files (Parquet)
├── features/           # Feature engineering modules
│   ├── elo.py          # ELO rating calculations
│   └── venue_stats.py  # Venue bias analysis
├── inference/          # Inference logic for live predictions
├── models/             # Trained ML models (.pkl)
└── notebooks/          # Jupyter notebooks for experimentation
```

## ▶️ How to Run

### Run the Streamlit App
```bash
streamlit run app.py
```

### Run the API Server
```bash
uvicorn api.main:app --reload
```
