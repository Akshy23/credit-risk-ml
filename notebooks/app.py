from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

artifact = joblib.load("credit_default_random_forest.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
features = artifact["features"]

app = FastAPI(title="Credit Default Prediction API")


class Applicant(BaseModel):
    person_age: float | None = None
    person_income: float
    person_home_ownership: Literal["MORTGAGE", "OTHER", "OWN", "RENT"]
    person_emp_length: float | None = None
    loan_intent: Literal[
        "DEBTCONSOLIDATION", "EDUCATION", "HOMEIMPROVEMENT",
        "MEDICAL", "PERSONAL", "VENTURE"
    ]
    loan_amnt: float
    loan_percent_income: float
    cb_person_default_on_file: Literal["Y", "N"]
    cb_person_cred_hist_length: float


@app.post("/predict")
def predict_default(applicant: Applicant):
    applicant_df = pd.DataFrame(
        [applicant.model_dump()],
        columns=features
    )

    probability = float(model.predict_proba(applicant_df)[0, 1])
    prediction = (
        "Default" if probability >= threshold else "No default"
    )

    return {
        "default_probability": round(probability, 4),
        "prediction": prediction,
        "threshold": threshold,
    }
