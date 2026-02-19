import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health AI", layout="wide")

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #f5fff7, #e8f5e9);
}

.navbar {
    display:flex;
    justify-content:space-between;
    align-items:center;
    padding:20px 40px;
    background:white;
    border-radius:12px;
    margin-bottom:20px;
    box-shadow:0 4px 20px rgba(0,0,0,0.05);
}

.logo {
    font-size:24px;
    font-weight:700;
    color:#2E7D32;
}

.hero-title {
    font-size:52px;
    font-weight:700;
    color:#222;
}

.hero-title span {
    color:#2E7D32;
}

.hero-text {
    font-size:18px;
    color:#555;
}

.btn {
    background:linear-gradient(45deg,#2E7D32,#66BB6A);
    color:white;
    padding:12px 26px;
    border-radius:10px;
    border:none;
    font-weight:600;
}

.card {
    background:white;
    padding:25px;
    border-radius:16px;
    box-shadow:0 10px 25px rgba(0,0,0,0.05);
    text-align:center;
}

</style>
""", unsafe_allow_html=True)

# ---------- NAVBAR ----------
st.markdown("""
<div class="navbar">
    <div class="logo">🌿 HealthAI</div>
    <div>Home &nbsp;&nbsp; About &nbsp;&nbsp; Features &nbsp;&nbsp; Contact</div>
</div>
""", unsafe_allow_html=True)

# ---------- Load dataset ----------
df = pd.read_excel("health_prediction_enhanced_500.xlsx")

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

# ---------- NAV ----------
menu = st.sidebar.radio("Navigation", ["Home", "Questionnaire", "Prediction"])

# ---------- HOME PAGE ----------
if menu == "Home":

    col1, col2 = st.columns([1.3,1])

    with col1:
        st.markdown('<p class="hero-title">AI Powered <span>Health Prediction</span></p>', unsafe_allow_html=True)
        st.markdown('<p class="hero-text">Answer a few questions and get instant AI insights about your health status.</p>', unsafe_allow_html=True)

        st.markdown("###")
        st.button("🚀 Get Started")
        st.button("📖 Learn More")

    with col2:
        st.image("https://images.unsplash.com/photo-1505751172876-fa1923c5c528")

    st.markdown("##")

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="card">⚡<h4>Instant Prediction</h4><p>Get results in seconds</p></div>', unsafe_allow_html=True)
    c2.markdown('<div class="card">📊<h4>Personal Insights</h4><p>AI powered analysis</p></div>', unsafe_allow_html=True)
    c3.markdown('<div class="card">💚<h4>User Friendly</h4><p>Simple interface</p></div>', unsafe_allow_html=True)

# ---------- QUESTIONNAIRE ----------
elif menu == "Questionnaire":

    st.title("Health Questionnaire 📝")

    age = st.slider("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male","Female"])
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    cholesterol = st.slider("Cholesterol", 100, 350, 200)
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)

    smoking = st.radio("Do you smoke?", ["No","Yes"])
    exercise = st.selectbox("Exercise Level", ["Low","Moderate","High"])

    fever = st.radio("Fever?", ["No","Yes"])
    cough = st.radio("Cough?", ["No","Yes"])
    headache = st.radio("Headache?", ["No","Yes"])
    fatigue = st.radio("Fatigue?", ["No","Yes"])

    risk_score = st.slider("Health Feeling Score", 0.0, 1.0, 0.5)

    yes_no = lambda x: 1 if x=="Yes" else 0

    st.session_state["input_data"] = {
        "fever": yes_no(fever),
        "cough": yes_no(cough),
        "headache": yes_no(headache),
        "fatigue": yes_no(fatigue),
        "age": age,
        "bp": bp,
        "cholesterol": cholesterol,
        "gender": gender,
        "bmi": bmi,
        "smoking": yes_no(smoking),
        "exercise_level": exercise,
        "sleep_hours": sleep,
        "risk_score": risk_score
    }

    st.success("✅ Answers saved! Go to Prediction")

# ---------- PREDICTION ----------
elif menu == "Prediction":

    st.title("Prediction 🔍")

    if "input_data" not in st.session_state:
        st.warning("Fill questionnaire first")
    else:
        data = st.session_state["input_data"]

        gender = le_gender.transform([data["gender"]])[0]
        exercise = le_exercise.transform([data["exercise_level"]])[0]

        input_array = np.array([[data["fever"], data["cough"], data["headache"], data["fatigue"],
                                 data["age"], data["bp"], data["cholesterol"],
                                 gender, data["bmi"], data["smoking"],
                                 exercise, data["sleep_hours"], data["risk_score"]]])

        prediction = model.predict(input_array)
        disease = le_disease.inverse_transform(prediction)[0]

        st.success(f"🩺 Predicted Health Status: {disease}")
