import pickle
import pandas as pd
from pydantic import BaseModel

from fastapi import FastAPI


# Pydantic classes for input and output
class PatientInformation(BaseModel):
    id: int
    gender: str
    age: float
    hypertension: int
    heart_disease: int
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str


class PredictionOut(BaseModel):
    default_proba: float

# Load the model
model = pickle.load(open('lgbm_pipeline-0.1.pkl', 'rb'))

# Start the app
app = FastAPI()

# Home page
@app.get("/")
def home():
    return {"message": "Risk of Stroke Prediction App", "model_version": 0.1}

# Inference endpoint
@app.post("/predict", response_model=PredictionOut)
def predict(payload: PatientInformation):
    cust_df = pd.DataFrame([payload.dict()])
    preds = model.predict_proba(cust_df)[0, 1]
    result = {"default_proba": preds}
    return result
