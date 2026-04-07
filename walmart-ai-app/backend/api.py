from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import pandas as pd
import numpy as np
import os

# --------------------------------------------------
# Create FastAPI App
# --------------------------------------------------

app = FastAPI(title="Walmart Sales Prediction API")


# --------------------------------------------------
# Enable CORS (Important for GitHub Pages frontend)
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # allow requests from any frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# Load Model
# --------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model1.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)


# --------------------------------------------------
# Root Endpoint
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Walmart Sales Prediction API is running"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: list):

    try:

        df = pd.DataFrame(data)

        # Convert numeric columns safely
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        predictions = model.predict(df)

        df["prediction"] = predictions

        return df.to_dict(orient="records")

    except Exception as e:

        return {
            "error": str(e)
        }
