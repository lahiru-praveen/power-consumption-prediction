from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- 1. NEW IMPORT
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI(title="Phase Power Consumption Predictor")

# This tells the API to accept requests from outside domains.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # The "*" means "allow requests from ANY website".
    allow_credentials=True,
    allow_methods=["*"],  # Allows all types of requests (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the trained model at startup
try:
    model = joblib.load("hourly_phase_model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# Define the expected JSON payload schema
class PredictionRequest(BaseModel):
    timestamp: str
    phase_a_lags: list[float]
    phase_b_lags: list[float]
    phase_c_lags: list[float]

@app.post("/predict")
def predict_next_hour(data: PredictionRequest):
    if len(data.phase_a_lags) != 24 or len(data.phase_b_lags) != 24 or len(data.phase_c_lags) != 24:
        raise HTTPException(status_code=400, detail="Each phase must contain exactly 24 lagging hours of data.")

    try:
        dt = datetime.strptime(data.timestamp, "%Y-%m-%d %H:%M:%S")
        current_hour = dt.hour
        day_of_week = dt.weekday()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS'.")

    features = {}

    for i in range(1, 25):
        features[f'Phase_A_kW_lag_{i}'] = data.phase_a_lags[i - 1]
    for i in range(1, 25):
        features[f'Phase_B_kW_lag_{i}'] = data.phase_b_lags[i - 1]
    for i in range(1, 25):
        features[f'Phase_C_kW_lag_{i}'] = data.phase_c_lags[i - 1]

    features['hour'] = current_hour
    features['day_of_week'] = day_of_week

    input_df = pd.DataFrame([features])
    prediction = model.predict(input_df)[0]

    return {
        "target_hour": f"{(current_hour + 1) % 24}:00",
        "predicted_Phase_A_kW": float(prediction[0]),
        "predicted_Phase_B_kW": float(prediction[1]),
        "predicted_Phase_C_kW": float(prediction[2])
    }

@app.get("/health")
def health_check():
    return {"status": "healthy"}