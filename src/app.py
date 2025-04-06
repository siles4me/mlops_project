import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("churn_api")

# Configuración desde variables de entorno
MODEL_PATH = os.getenv("MODEL_PATH", "models/churn_model_random_forest.joblib")
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

# Carga del modelo
model = joblib.load(MODEL_PATH)

# Inicializar FastAPI
app = FastAPI(title="Churn Prediction API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict")
def predict_churn(payload: dict):
    try:
        df = pd.DataFrame([payload])
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0, 1]
        return {"prediction": int(pred), "probability": float(prob)}
    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        raise HTTPException(status_code=400, detail=str(e))
