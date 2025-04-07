import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------
# Cargar datos
# ----------------------
csv_path = Path("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = pd.read_csv(csv_path)
df = df.drop("customerID", axis=1)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["SeniorCitizen"] = df["SeniorCitizen"].astype(str)
df.dropna(inplace=True)

target_col = "Churn"
numeric_cols = ["MonthlyCharges", "TotalCharges", "tenure"]
categorical_cols = [col for col in df.columns if col not in numeric_cols + [target_col]]

X = df.drop(columns=[target_col])
y = df[target_col].map({"No": 0, "Yes": 1})

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

param_grid = {
    "classifier__n_estimators": [100],
    "classifier__max_depth": [10, 20, None],
    "classifier__min_samples_split": [2],
    "classifier__min_samples_leaf": [1],
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gs = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring="accuracy")
gs.fit(X_train, y_train)

best_model = gs.best_estimator_
y_pred = best_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"✅ Accuracy: {acc*100:.2f}%")
print(report)

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)

joblib.dump(best_model, model_dir / "churn_model.joblib")
joblib.dump(preprocessor, model_dir / "encoder.joblib")
joblib.dump(X.columns.tolist(), model_dir / "expected_columns.joblib")

with open("classification_report.txt", "w") as f:
    f.write(report)

with open("metrics.txt", "w") as f:
    f.write(f"Accuracy: {acc*100:.2f}%\n")
    f.write(report)

# Importancia de características
feature_names = best_model.named_steps["preprocessor"].get_feature_names_out()
importances = best_model.named_steps["classifier"].feature_importances_

plt.figure(figsize=(10, 8))
plt.barh(feature_names, importances, color="skyblue")
plt.xlabel("Importancia")
plt.title("Feature Importance - RandomForest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

print("✅ Entrenamiento y guardado finalizados.")