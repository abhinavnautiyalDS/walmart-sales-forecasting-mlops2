from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import pickle
import os

# ---------------------------------------------------
# Initialize FastAPI
# ---------------------------------------------------

app = FastAPI(title="Walmart Sales Prediction API")

# ---------------------------------------------------
# Enable CORS (required for GitHub Pages frontend)
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Load trained model
# ---------------------------------------------------

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model1.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

print("Model loaded successfully")

# ---------------------------------------------------
# Health check endpoint
# ---------------------------------------------------

@app.get("/")
def home():
    return {"message": "Walmart Sales Prediction API running"}

# ---------------------------------------------------
# Prediction endpoint
# ---------------------------------------------------

@app.post("/predict")
async def predict(request: Request):

    try:

        # Receive JSON from frontend
        data = await request.json()

        if not isinstance(data, list):
            return {"error": "Input must be a list of records"}

        df = pd.DataFrame(data)

        # Convert numeric columns
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")

        # Run model prediction
        predictions = model.predict(df)

        df["prediction"] = predictions

        return df.to_dict(orient="records")

    except Exception as e:

        return {"error": str(e)}
