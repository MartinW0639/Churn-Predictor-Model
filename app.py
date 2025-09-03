# Churn 1-> Yes, 0 -> No
# Gender 1-> Female, 0-> Male
# Scaler exported as scaler.pkl
# Model exported as model.pkl
# Order of the X -> "Age", "Gender", "Tenure", "MonthlyCharges"

import streamlit as st
import joblib
import numpy as np


scaler = joblib.load("scaler.pkl")
model = joblib.load("model.pkl")

st.title("Churn Predictor App")

st.divider()

st.write("Please enter your values and hit the predict button")

st.divider()

age = st.number_input("Enter age", min_value=10, max_value=100, value=30)

tenure = st.number_input("Enter Tenure",  min_value=0, max_value=130, value=10)

monthlyCharge = st.number_input("Enter Monthly Charge", min_value=30, max_value=150)

gender = st.selectbox("Enter the Gender", ["Male", "Female"])

st.divider()

predictbutton = st.button("Predict")

if predictbutton:
    gender_selected = 1 if gender == "Female" else 0
    X=[age, gender_selected, tenure, monthlyCharge]

    X1 = np.array(X)
    X_array = scaler.transform([X1])

    prediction = model.predict(X_array)[0]

    predicted = "Churn" if prediction == 1 else "Not Churn"

    st.write(f"Predicted: {predicted}")
 
else: 
    print("Please enter the values and press the predict button")
