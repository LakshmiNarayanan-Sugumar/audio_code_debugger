import numpy as np
import yfinance as yf
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model

LOOKBACK = 60
TICKER   = "TMPV.NS"

model  = load_model("tata_motors_bilstm.keras")
scaler = joblib.load("data/scaler_y.pkl")

app = FastAPI()

class PredictResponse(BaseModel):
    ticker: str
    current_price: float
    predicted_price: float
    currency: str

@app.get("/")
def home():
    return {"message": "Tata Motors Stock Predictor API is running"}

@app.post("/predict", response_model=PredictResponse)
def predict():
    # fetching data fresh on every request
    df = yf.download(TICKER, period="6mo", progress=False)

    if df.empty:
        raise HTTPException(status_code=500, detail="No data available")

    close_prices  = df["Close"].values.flatten()[-LOOKBACK:]
    current_price = float(df["Close"].values.flatten()[-1])

    scaled  = scaler.transform(close_prices.reshape(-1, 1))
    X_input = scaled.reshape(1, LOOKBACK, 1)

    prediction_scaled = model.predict(X_input)
    predicted_price   = float(scaler.inverse_transform(prediction_scaled)[0][0])

    return PredictResponse(
        ticker=TICKER,
        current_price=round(current_price, 2),
        predicted_price=round(predicted_price, 2),
        currency="INR"
    )