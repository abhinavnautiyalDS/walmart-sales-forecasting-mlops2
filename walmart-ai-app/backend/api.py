from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
import os

# Create FastAPI app
app = FastAPI()

# Load model safely
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model1.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.get("/")
def home():
    return {"message": "Walmart Sales Prediction API Running"}

@app.post("/predict")
def predict(data: list):

    df = pd.DataFrame(data)

    predictions = model.predict(df)

    df["prediction"] = predictions

    return df.to_dict(orient="records")