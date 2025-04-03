import streamlit as st
import requests

# Título de la app
st.title("Predicción de Churn")

# Crear un formulario para ingresar los datos del cliente
st.header("Datos del cliente")

tenure = st.number_input("Tenure (Meses)", min_value=0)
monthly_charges = st.number_input("Monthly Charges (USD)", min_value=0.0)
total_charges = st.number_input("Total Charges (USD)", min_value=0.0)
gender = st.selectbox("Gender", options=["Female", "Male"])
senior_citizen = st.selectbox("Senior Citizen", options=[0, 1])
partner = st.selectbox("Partner", options=["Yes", "No"])
dependents = st.selectbox("Dependents", options=["Yes", "No"])
phone_service = st.selectbox("Phone Service", options=["Yes", "No"])
multiple_lines = st.selectbox("Multiple Lines", options=["Yes", "No", "No phone service"])
internet_service = st.selectbox("Internet Service", options=["DSL", "Fiber optic", "No"])
online_security = st.selectbox("Online Security", options=["Yes", "No", "No internet service"])
online_backup = st.selectbox("Online Backup", options=["Yes", "No", "No internet service"])
device_protection = st.selectbox("Device Protection", options=["Yes", "No", "No internet service"])
tech_support = st.selectbox("Tech Support", options=["Yes", "No", "No internet service"])
streaming_tv = st.selectbox("Streaming TV", options=["Yes", "No", "No internet service"])
streaming_movies = st.selectbox("Streaming Movies", options=["Yes", "No", "No internet service"])
contract = st.selectbox("Contract", options=["Month-to-month", "One year", "Two year"])
paperless_billing = st.selectbox("Paperless Billing", options=["Yes", "No"])
payment_method = st.selectbox("Payment Method", options=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

# Botón para hacer la predicción
if st.button("Predecir Churn"):
    # Organizar los datos en un diccionario
    data = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method
    }

    # Realizar la solicitud POST a la API
    response = requests.post("http://127.0.0.1:8000/predict", json=data)

    # Mostrar los resultados
    if response.status_code == 200:
        result = response.json()
        st.write(f"Predicción de Churn: {'Sí' if result['prediction'] == 1 else 'No'}")
        st.write(f"Probabilidad de Churn: {result['churn_probability']:.2f}")
    else:
        st.write("Error al realizar la predicción.")