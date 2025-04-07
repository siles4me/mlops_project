# 📊 Modelo de Predicción de Churn de Clientes

Este proyecto implementa un modelo de *machine learning* para predecir si un cliente abandonará el servicio (*churn*). Utiliza un pipeline de procesamiento de datos con `scikit-learn`, un modelo de `RandomForestClassifier` y una API desarrollada con `FastAPI` para servir predicciones.

---

## 📁 Estructura del Proyecto

```
mlops_project/
├── .github/
│   └── workflows/
│       └── main.yml                  # Workflow CI/CD para entrenamiento, build y despliegue
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset original (solo lectura)
├── models/
│   ├── churn_model.joblib            # Modelo entrenado 
│   ├── encoder.joblib               # Preprocesador ColumnTransformer 
│   └── expected_columns.joblib      # Columnas esperadas para predicción
├── notebooks/
│   └── limpieza_y_exploracion.ipynb  # Notebook de limpieza y análisis exploratorio
├── src/
│   ├── app.py                       # API con FastAPI para predicción
│   ├── train_model.py               # Script de entrenamiento y reentrenamiento
│   └── app_ui.py.py                 # app de ejecucion de streamlit 
├── .dockerignore                    # Archivos/carpetas excluidos del build Docker
├── .gitignore                       # Archivos/carpetas excluidos de Git
├── Dockerfile                       # Multistage build para la API
├── requirements.txt                 # Dependencias de producción
├── requirements-dev.txt             # Dependencias de desarrollo (notebooks, tests)
└── README.md                        # Documentación del proyecto
```

---

## 🚀 1. Entrenamiento del Modelo

### ✅ Requisitos
Instalar las dependencias con el siguiente comando:

```bash
pip install -r requirements.txt
```

### ▶️ Ejecutar el script de entrenamiento
Para entrenar el modelo y guardar los artefactos generados, ejecuta el siguiente comando:

```bash
python train_model.py
```

Este script realizará las siguientes acciones:

- Limpia y transforma los datos.
- Entrena un modelo utilizando validación cruzada (*GridSearchCV*).
- Guarda el modelo entrenado (`churn_model.joblib`), el preprocesador (`encoder.joblib`) y las columnas esperadas (`expected_columns.joblib`).
- Genera reportes de clasificación y un gráfico de importancia de variables.

## 🌐 2. API de Predicción con FastAPI

### ▶️ Correr el servidor
Para ejecutar la API, utiliza el siguiente comando:

```bash
uvicorn app:app --reload
```

### 📥 Endpoint de predicción
La API tiene el siguiente endpoint para realizar predicciones:

`POST /api/v1/predict-churn`

### 🧾 Ejemplo de entrada
Envía una solicitud con el siguiente formato de entrada:

```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 5,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "DSL",
  "OnlineSecurity": "Yes",
  "OnlineBackup": "No",
  "DeviceProtection": "Yes",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 350.5
}
```

### ✅ Ejemplo de respuesta

La respuesta será algo similar a esto:

```json
{
  "prediction": "Cliente propenso a Churn",
  "probabilidad_churn": "78.21%"
}
```

### 🛠️ Tecnologías Utilizadas
- Python 3.10+
- `scikit-learn`
- `pandas`, `numpy`
- `FastAPI`
- `Uvicorn`
- `joblib`
- `matplotlib`

### 📌 Notas Importantes
- El preprocesamiento de variables categóricas se realiza con `OneHotEncoder` dentro de un pipeline de `ColumnTransformer`, lo que asegura la consistencia entre entrenamiento y predicción.
- Las columnas del input deben coincidir en nombre y orden con las esperadas por el modelo. Esto se controla con el archivo `expected_columns.joblib`.

### 🌐 API en Producción

La API de predicción está desplegada y accesible en la siguiente URL, donde puedes consultar la documentación interactiva de los endpoints disponibles:

[Documentación de la API](http://ec2-18-218-76-254.us-east-2.compute.amazonaws.com:8000/docs)

### ✍️ Autor
David Siles — [LinkedIn](https://www.linkedin.com/in/david-siles-865130182)
