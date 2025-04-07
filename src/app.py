from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, conint, confloat
from typing import Literal
import pandas as pd
import joblib
from pathlib import Path
import uvicorn

app = FastAPI(
    title="Churn Prediction API",
    version="1.0.0",
    description="Modelo predictivo para determinar si un cliente hará churn"
)

MODEL_PATH = Path("models/churn_model.joblib")
COLUMNS_PATH = Path("models/expected_columns.joblib")

try:
    model = joblib.load(MODEL_PATH)
    expected_columns = joblib.load(COLUMNS_PATH)
except Exception as e:
    raise RuntimeError(f"Error loading model or expected columns: {e}")

class ChurnInput(BaseModel):
    gender: Literal["Female", "Male"]
    SeniorCitizen: conint(ge=0, le=1)
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: conint(ge=0)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: confloat(gt=0)
    TotalCharges: confloat(gt=0)

    @field_validator("TotalCharges")
    def check_total_charges(cls, value):
        if value < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="TotalCharges debe ser mayor que cero."
            )
        return value

@app.post("/api/v1/predict-churn", tags=["churn"])
async def predict_churn(data: ChurnInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

        # Aseguramos orden correcto
        df = df.reindex(columns=expected_columns)

        # Predicción
        prediction = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        churn_label = "Cliente propenso a Churn" if prediction == 1 else "Cliente NO propenso a Churn"

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "prediction": churn_label,
                "probabilidad_churn": f"{proba:.2%}"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error interno: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
