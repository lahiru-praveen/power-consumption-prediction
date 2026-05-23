from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
from datetime import datetime

app = FastAPI(title="Phase Power Consumption Predictor")

# Load the trained model at startup
try:
    model = joblib.load("hourly_phase_model.pkl")
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")


# Define the expected JSON payload schema
class PredictionRequest(BaseModel):
    timestamp: str  # Format: "YYYY-MM-DD HH:MM:SS" or ISO format
    phase_a_lags: list[float]  # Must contain exactly 24 values (t-1 down to t-24)
    phase_b_lags: list[float]  # Must contain exactly 24 values
    phase_c_lags: list[float]  # Must contain exactly 24 values


@app.post("/predict")
def predict_next_hour(data: PredictionRequest):
    # Validation: Ensure exactly 24 lags are provided for each phase
    if len(data.phase_a_lags) != 24 or len(data.phase_b_lags) != 24 or len(data.phase_c_lags) != 24:
        raise HTTPException(status_code=400, detail="Each phase must contain exactly 24 lagging hours of data.")

    try:
        # Preprocessing: Parse current request timestamp for temporal patterns
        dt = datetime.strptime(data.timestamp, "%Y-%m-%d %H:%M:%S")
        current_hour = dt.hour
        day_of_week = dt.weekday()
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format. Use 'YYYY-MM-DD HH:MM:SS'.")

    # Reconstruct the exact feature row structure your model was trained on
    # Order: phase_a_lag_1...24, phase_b_lag_1...24, phase_c_lag_1...24, hour, day_of_week
    features = {}

    for i in range(1, 25):
        features[f'Phase_A_kW_lag_{i}'] = data.phase_a_lags[i - 1]
    for i in range(1, 25):
        features[f'Phase_B_kW_lag_{i}'] = data.phase_b_lags[i - 1]
    for i in range(1, 25):
        features[f'Phase_C_kW_lag_{i}'] = data.phase_c_lags[i - 1]

    features['hour'] = current_hour
    features['day_of_week'] = day_of_week

    # Convert dictionary to DataFrame (1 row) matching training feature order
    input_df = pd.DataFrame([features])

    # Run the prediction
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