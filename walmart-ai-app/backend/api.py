from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np

app = FastAPI()

# Load model
model = pickle.load(open("model1.pkl","rb"))

@app.get("/")
def home():
    return {"message":"Walmart AI API running"}

@app.post("/predict")
def predict(data: list):

    df = pd.DataFrame(data)

    preds = model.predict(df)

    df["prediction"] = preds

    return df.to_dict(orient="records")
