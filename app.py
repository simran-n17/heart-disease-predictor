import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model
model, scaler = pickle.load(open('model.pkl', 'rb'))

# UI
st.title("Heart Disease Prediction")
st.write("Enter the details below to check the risk of heart disease:")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)")
chol = st.number_input("Cholesterol (chol)")
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG Results (restecg)", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (thalach)")
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("Oldpeak (ST depression induced by exercise)")
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0–3) colored by fluoroscopy", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

if st.button("Predict"):
    user_data = np.array([[age, 1 if sex == "Male" else 0, cp, trestbps, chol, fbs, restecg,
                           thalach, exang, oldpeak, slope, ca, thal]])
    user_scaled = scaler.transform(user_data)
    prediction = model.predict(user_scaled)
    st.success("Prediction: " + ("🫀 Heart Disease Risk" if prediction[0] == 1 else "✅ No Heart Disease"))
