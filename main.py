from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="Iris Prediction API")

# Load trained model
model = joblib.load("model.pkl")

# Iris class labels
IRIS_CLASSES = {0: "setosa", 1: "versicolor", 2: "virginica"}

# Proper request schema with validation
class IrisInput(BaseModel):
    features: List[float] = Field(..., min_length=4, max_length=4, 
                                   example=[5.1, 3.5, 1.4, 0.2])

@app.get("/")
def home():
    return {"message": "Iris Prediction API Running"}

@app.post("/predict")
def predict(data: IrisInput):
    try:
        features = np.array(data.features).reshape(1, -1)
        prediction = model.predict(features)
        class_id = int(prediction[0])
        return {
            "prediction": class_id,
            "species": IRIS_CLASSES[class_id]  # Human-readable label
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))  # Proper error