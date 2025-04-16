import streamlit as st
import pickle
import numpy as np

# Load the model and scaler
model, scaler = pickle.load(open('model.pkl', 'rb'))

st.title("Heart Disease Prediction")

# Input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.number_input("Serum Cholestoral in mg/dl (chol)", value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.selectbox("Resting ECG (restecg)", [0, 1, 2])
thalch = st.number_input("Maximum Heart Rate (thalch)", value=150)
exang = st.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.number_input("ST Depression (oldpeak)", value=1.0)
slope = st.selectbox("Slope of the Peak (slope)", [0, 1, 2])
ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3])
thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

# Convert inputs
sex = 1 if sex == "Male" else 0

features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                      thalch, exang, oldpeak, slope, ca, thal]])

# Scale input
scaled_input = scaler.transform(features)

if st.button("Predict"):
    prediction = model.predict(scaled_input)
    result = "Heart Disease Detected" if prediction[0] == 1 else "No Heart Disease"
    st.success(result)
