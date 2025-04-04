# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# Cargar Dataset
csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)

# Preprocesamiento
# Eliminar 'customerID' ya que no aporta valor
df = df.drop(["customerID"], axis=1)

# Convertir 'TotalCharges' a numérico (algunos valores pueden ser no numéricos)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

# Convertir 'SeniorCitizen' a categórico
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)

# Aplicar One-Hot Encoding a variables categóricas
df = pd.get_dummies(df, drop_first=True)

# Manejar valores nulos (rellenamos con la mediana)
df.fillna(df.median(), inplace=True)

# Escalar variables numéricas
scaler = StandardScaler()
numeric_cols = ["MonthlyCharges", "TotalCharges", "tenure"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


# Variables predictoras (X) y objetivo (y)
X = df.drop(columns=["Churn_Yes"])  # 'Churn_Yes' es la variable objetivo tras One-Hot Encoding
y = df["Churn_Yes"]

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pipeline de Modelado
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Optimización de Hiperparámetros con GridSearchCV
param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

grid_search_rf = GridSearchCV(pipeline, param_grid_rf, cv=5, n_jobs=-1, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)

# Mejor Modelo y Evaluación
best_model = grid_search_rf.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Guardar reporte de clasificación
report = classification_report(y_test, y_pred)
with open("classification_report.txt", 'w') as f:
    f.write(report)

# Guardar métricas para CML
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy * 100:.2f}%\n")
    f.write(report)

# Visualización de la Importancia de las Características
importances = best_model.named_steps['classifier'].feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importancia')
plt.title('Importancia de las características en RandomForest')
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=120)
plt.close()

# Guardado del Modelo
model_path = Path("models/churn_model_random_forest.joblib")
joblib.dump(best_model, model_path)
print(f"✅ Modelo entrenado y guardado en '{model_path}'")
