from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel
from pathlib import Path

# Crear instancia de FastAPI
app = FastAPI()

# Definir la ruta del modelo entrenado
model_path = Path("models/churn_model_random_forest.joblib")


# Cargar el modelo entrenado
if not model_path.exists():
    raise Exception(f"El modelo no se encuentra en la ruta: {model_path}")

model = joblib.load(model_path)

# Obtener las columnas del modelo
expected_columns = model.feature_names_in_

# Definir la estructura de los datos de entrada
class CustomerFeatures(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str

# Endpoint de prueba
@app.get("/")
def home():
    return {"message": "API de Predicción de Churn Activa 🚀"}

# Función para preprocesar la entrada
def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza y transformación de datos antes de la predicción."""
    # Convertir TotalCharges a numérico (para evitar valores vacíos)
    data["TotalCharges"] = pd.to_numeric(data["TotalCharges"], errors="coerce").fillna(0)

    # Convertir variables categóricas a valores numéricos mediante One-Hot Encoding
    categorical_columns = ["gender", "Partner", "Dependents", "PhoneService",
                           "MultipleLines", "InternetService", "OnlineSecurity",
                           "OnlineBackup", "DeviceProtection", "TechSupport",
                           "StreamingTV", "StreamingMovies", "Contract",
                           "PaperlessBilling", "PaymentMethod"]

    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

    # Asegurarse de que las columnas coincidan con las esperadas por el modelo
    data = data.reindex(columns=expected_columns, fill_value=0)

    return data

# Endpoint para realizar predicciones
@app.post("/predict")
def predict_churn(features: CustomerFeatures):
    """ Realiza la predicción del churn para un cliente con las características dadas. """
    try:
        # Convertir JSON a DataFrame
        input_data = pd.DataFrame([features.dict()])

        # Aplicar las transformaciones necesarias
        input_data = preprocess_input(input_data)

        # Realizar la predicción
        prediction = model.predict(input_data)
        probability = model.predict_proba(input_data)[:, 1]

        return {
            "prediction": int(prediction[0]),  # 0 = No churn, 1 = Churn
            "churn_probability": float(probability[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))