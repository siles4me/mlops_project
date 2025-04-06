# MLOps Project: Customer Churn Prediction

## Descripción General
Este repositorio implementa un **pipeline completo de MLOps** para un modelo de predicción de churn de clientes en una compañía de telecomunicaciones. Incluye:

- **Entrenamiento y reentrenamiento** del modelo con optimización de hiperparámetros.
- **API REST** para predicción usando **FastAPI**.
- **Contenerización** con **Docker** (multistage build y healthcheck).
- **Integración Continua y Despliegue Continuo (CI/CD)** mediante **GitHub Actions**.
- **Gestión de ramas**: `dev`, `staging`, `main`.

## Estrategia de Ramas

| Rama      | Propósito                                      |
|-----------|------------------------------------------------|
| **dev**   | Desarrollo activo: aquí se integran nuevas features y cambios. Se ejecuta el workflow de CI/CD para pruebas y reentrenamiento. |
| **staging** | Pruebas de integración antes de producción. Se ejecutan PRs desde `dev` para validar el modelo y la API en un entorno staging. |
| **main**  | Código estable en producción. Los cambios solo llegan tras validación en `staging`. Aquí corre el despliegue final. |

## Estructura del Proyecto

```bash
mlops_project/
├── .github/
│   └── workflows/
│       └── main.yml               # Workflow CI/CD para entrenamiento, build y despliegue
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Dataset original (solo lectura)
├── models/
│   └── churn_model_random_forest.joblib      # Modelo entrenado y versionado
├── notebooks/
│   └── limpieza_y_exploracion.ipynb           # Notebook de limpieza y análisis exploratorio
├── src/
│   ├── app.py                               # API con FastAPI para predicción
│   ├── train_model.py                       # Script de entrenamiento y reentrenamiento
│   └── utils.py                             # Funciones auxiliares (si aplican)
├── .dockerignore                            # Archivos/carpetas excluidos del build Docker
├── .gitignore                               # Archivos/carpetas excluidos de Git
├── Dockerfile                               # Multistage build para la API
├── requirements.txt                         # Dependencias de producción
├── requirements-dev.txt                     # Dependencias de desarrollo (notebooks, tests)
└── README.md                                # Documentación del proyecto
```

## Descripción de Archivos y Carpetas

### `.github/workflows/main.yml`
- **Pipeline de CI/CD**:
  1. **Entrenamiento**: ejecuta `src/train_model.py` dentro de un contenedor `dvcorg/cml:latest` para reentrenar el modelo.
  2. **Reporte CML**: genera `metrics.txt` y `feature_importance.png` y publica un comentario en PR.
  3. **Build Docker**: construye la imagen `DOCKER_USERNAME/churn-api:dev`.
  4. **Push Docker**: sube la imagen a DockerHub.
  5. **Despliegue EC2**: conecta por SSH a la instancia EC2 y actualiza el contenedor `churn-api`.

### `data/`  
- Contiene los datos crudos. El script `src/train_model.py` los carga para limpieza y entrenamiento. Nunca se modifican directamente.

### `models/`  
- Almacena los artefactos de modelo: `churn_model_random_forest.joblib`.


### `src/train_model.py`
- **Carga**: lee `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Preprocesamiento**: drop de `customerID`, conversión de tipos, One‑Hot Encoding, manejo de nulos, escalado.
- **División**: `train_test_split`.
- **Pipeline**: `StandardScaler` + `RandomForestClassifier`.
- **GridSearchCV**: optimización de hiperparámetros.
- **Evaluación**: guarda `classification_report.txt`, `metrics.txt`.
- **Visualización**: grafica `feature_importance.png`.
- **Modelo**: guarda `models/churn_model_random_forest.joblib`.

### `src/app.py`
- **FastAPI**:
  - `/health`: endpoint de salud.
  - `/predict`: recibe JSON con características del cliente, retorna predicción y probabilidad.
- **Configuración**:
  - Lee `MODEL_PATH` y `CORS_ORIGINS` desde variables de entorno.
  - Logging a nivel INFO.
  - CORS habilitado para dominios configurables.

### `Dockerfile`
- **Multistage build**:
  1. **builder**: instala dependencias en `/install`.
  2. **final**: copia dependencias, código y modelo.
- **Healthcheck**: verifica `/health` cada 30s.
- **CMD**: arranca FastAPI con `uvicorn src.app:app`.

### `requirements.txt` y `requirements-dev.txt`
- `requirements.txt`: dependencias mínimas para producción.
- `requirements-dev.txt`: herramientas de desarrollo (Jupyter, tests, linting).

### `.gitignore` y `.dockerignore`
- `.gitignore`: excluye entornos, llaves, IDEs, notebooks.
- `.dockerignore`: además excluye datos crudos y notebooks del build.

