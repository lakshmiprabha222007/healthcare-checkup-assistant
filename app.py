import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health AI", page_icon="🌿", layout="wide")

# ---------- Modern CSS ----------
st.markdown("""
<style>

html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

.main {
    background: linear-gradient(135deg, #e8f1ff 0%, #f9fcff 100%);
}

.hero {
    text-align:center;
    padding:40px 20px;
}

.hero h1 {
    font-size:48px;
    font-weight:700;
    color:#1f3c88;
}

.hero p {
    font-size:20px;
    color:#555;
}

.glass {
    background: rgba(255,255,255,0.7);
    backdrop-filter: blur(12px);
    border-radius:20px;
    padding:30px;
    box-shadow:0 8px 32px rgba(31, 38, 135, 0.1);
    margin-bottom:25px;
}

.section-title {
    font-size:26px;
    font-weight:600;
    margin-bottom:15px;
    color:#1f3c88;
}

.stButton>button {
    background: linear-gradient(90deg,#1f6fff,#3ea6ff);
    color:white;
    border:none;
    border-radius:12px;
    height:45px;
    width:220px;
    font-size:16px;
    font-weight:600;
}

.stButton>button:hover {
    transform: scale(1.05);
    transition:0.2s;
}

.result-box {
    text-align:center;
    padding:40px;
    font-size:24px;
    font-weight:600;
    color:#1f3c88;
}

</style>
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

# ---------- Session ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ================= HOME =================
if st.session_state.page == "home":

    st.markdown("""
    <div class="hero">
        <h1>🌿 AI Health Risk Predictor</h1>
        <p>Smart AI powered health insights in seconds</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="glass">
        <h3>💚 Welcome to your personal AI health assistant</h3>
        <p>
        This intelligent system analyzes your health indicators and predicts possible risks.
        Start your assessment to get instant insights.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("🚀 Start Health Assessment"):
        st.session_state.page = "questionnaire"
        st.rerun()

# ================= QUESTIONNAIRE =================
elif st.session_state.page == "questionnaire":

    st.markdown('<div class="section-title">📝 Health Questionnaire</div>', unsafe_allow_html=True)
    st.markdown('<div class="glass">', unsafe_allow_html=True)

    st.subheader("Basic Information")
    age = st.slider("Age", 10, 80, 25)
    gender = st.selectbox("Gender", ["Male", "Female"])

    st.subheader("Health Metrics")
    bmi = st.slider("BMI", 15.0, 40.0, 22.0)
    bp = st.slider("Blood Pressure", 80, 180, 120)
    cholesterol = st.slider("Cholesterol", 100, 350, 200)
    sleep = st.slider("Sleep Hours", 3.0, 10.0, 7.0)

    st.subheader("Lifestyle")
    smoking = st.radio("Do you smoke?", ["No", "Yes"])
    exercise = st.selectbox("Exercise Level", ["Low", "Moderate", "High"])

    st.subheader("Symptoms")
    fever = st.radio("Fever?", ["No", "Yes"])
    cough = st.radio("Cough?", ["No", "Yes"])
    headache = st.radio("Headache?", ["No", "Yes"])
    fatigue = st.radio("Fatigue?", ["No", "Yes"])

    risk_score = st.slider("Overall Health Risk Feeling", 0.0, 1.0, 0.5)

    st.markdown('</div>', unsafe_allow_html=True)

    yes_no = lambda x: 1 if x == "Yes" else 0

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

    col1, col2 = st.columns(2)

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        if st.button("🔍 Predict Now"):
            st.session_state.page = "prediction"
            st.rerun()

# ================= PREDICTION =================
elif st.session_state.page == "prediction":

    st.markdown('<div class="section-title">🔍 Prediction Result</div>', unsafe_allow_html=True)

    if "input_data" not in st.session_state:
        st.warning("Please fill the questionnaire first")

    else:
        data = st.session_state["input_data"]

        gender = le_gender.transform([data["gender"]])[0]
        exercise = le_exercise.transform([data["exercise_level"]])[0]

        input_array = np.array([[
            data["fever"], data["cough"], data["headache"], data["fatigue"],
            data["age"], data["bp"], data["cholesterol"],
            gender, data["bmi"], data["smoking"],
            exercise, data["sleep_hours"], data["risk_score"]
        ]])

        prediction = model.predict(input_array)
        disease = le_disease.inverse_transform(prediction)[0]

        st.markdown(f"""
        <div class="glass result-box">
        🩺 Predicted Health Status: {disease}
        <br><br>
        <span style="font-size:16px;color:#555;">
        This is an AI prediction and not a medical diagnosis.
        </span>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Start New Assessment"):
            st.session_state.page = "home"
            st.rerun()

st.markdown("---")
st.caption("🌿 Health AI • Modern UI with Streamlit")
