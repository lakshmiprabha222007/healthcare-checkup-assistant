import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health Risk Predictor", layout="centered")

# ---------- Title ----------
st.markdown(
    "<h1 style='text-align: center; color:#4CAF50;'>🌿 Health Risk Predictor</h1>",
    unsafe_allow_html=True
)

st.write("### Enter your health details below")

# ---------- Sidebar image ----------
st.sidebar.image(
    "https://images.unsplash.com/photo-1505751172876-fa1923c5c528",
    caption="Stay Healthy 💚"
)

# ---------- Load dataset ----------
df = pd.read_excel("health_prediction_enhanced_500.xlsx")

# Encode categorical columns
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender'])

le_exercise = LabelEncoder()
df['exercise_level'] = le_exercise.fit_transform(df['exercise_level'])

le_disease = LabelEncoder()
df['disease'] = le_disease.fit_transform(df['disease'])

X = df.drop("disease", axis=1)
y = df["disease"]

model = RandomForestClassifier()
model.fit(X, y)

# ---------- User Inputs ----------
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 10, 80, 25)
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    cholesterol = st.slider("Cholesterol", 100, 350, 200)

with col2:
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)
    smoking = st.selectbox("Smoking", [0,1])
    gender = st.selectbox("Gender", ["Male","Female"])
    exercise = st.selectbox("Exercise Level", ["Low","Moderate","High"])

fever = st.selectbox("Fever", [0,1])
cough = st.selectbox("Cough", [0,1])
headache = st.selectbox("Headache", [0,1])
fatigue = st.selectbox("Fatigue", [0,1])
risk_score = st.slider("Risk Score", 0.0, 1.0, 0.5)

# Encode inputs
gender = le_gender.transform([gender])[0]
exercise = le_exercise.transform([exercise])[0]

input_data = np.array([[fever,cough,headache,fatigue,age,bp,cholesterol,
                        gender,bmi,smoking,exercise,sleep,risk_score]])

# ---------- Prediction ----------
if st.button("🔍 Predict Health Risk"):
    prediction = model.predict(input_data)
    disease = le_disease.inverse_transform(prediction)[0]

    st.success(f"### 🩺 Predicted Result: **{disease}**")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
