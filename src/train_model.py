import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Carga y preprocesamiento ---
csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)
df = df.drop("customerID", axis=1)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
df = pd.get_dummies(df, drop_first=True)
df.fillna(df.median(), inplace=True)
scaler = StandardScaler()
numeric_cols = ["MonthlyCharges", "TotalCharges", "tenure"]
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# --- División de datos ---
X = df.drop(columns=["Churn_Yes"])
y = df["Churn_Yes"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Pipeline y GridSearch ---
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(random_state=42))
])
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2, 5],
    "classifier__min_samples_leaf": [1, 2],
}
gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
gs.fit(X_train, y_train)

# --- Evaluación ---
best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc*100:.2f}%")
print(report)

# Guardar classification report
with open("classification_report.txt", "w") as f:
    f.write(report)

# Guardar métricas para CML
with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(report)

# Importancia de características
importances = best_model.named_steps["classifier"].feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color="skyblue")
plt.xlabel("Importancia")
plt.title("Feature Importance - RandomForest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

# Guardar modelo
model_path = Path("models/churn_model_random_forest.joblib")
joblib.dump(best_model, model_path)
print(f"✅ Modelo guardado en '{model_path}'")
