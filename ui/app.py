import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = joblib.load("models/heart_disease_model.pkl")

st.set_page_config(page_title="Heart Disease Prediction", layout="centered")

# Title
st.title("❤️ Heart Disease Prediction App")
st.write("Enter patient health data below and get a prediction in real-time.")

# User input form
age = st.number_input("Age", min_value=20, max_value=100, value=50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", min_value=80, max_value=200, value=120)
chol = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", min_value=70, max_value=220, value=150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST depression induced by exercise", min_value=0.0, max_value=10.0, value=1.0)
slope = st.selectbox("Slope of the peak exercise ST segment", [0, 1, 2])
ca = st.selectbox("Number of major vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (1 = normal; 2 = fixed defect; 3 = reversible defect)", [1, 2, 3])

# Convert inputs into DataFrame for prediction
input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]],
                          columns=['age', 'sex', 'cp', 'trestbps', 'chol',
                                   'fbs', 'restecg', 'thalach', 'exang',
                                   'oldpeak', 'slope', 'ca', 'thal'])

# Prediction
if st.button("🔍 Predict"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Heart Disease (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Low Risk of Heart Disease (Probability: {probability:.2f})")

# Bonus: Visualization (example with sample dataset)
st.subheader("📊 Heart Disease Trends")

# Load your cleaned dataset for visualization
try:
    df = pd.read_csv("cleaned_heart.csv")  # replace with your processed dataset
    fig, ax = plt.subplots()
    sns.countplot(x="target", data=df, ax=ax, palette="Set2")
    ax.set_title("Heart Disease Distribution")
    st.pyplot(fig)
except:
    st.info("Upload cleaned_heart.csv to see data visualization.")
