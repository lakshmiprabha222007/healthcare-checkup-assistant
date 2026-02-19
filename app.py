import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health AI", page_icon="🌿", layout="wide")

# ---------- Custom CSS ----------
st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}

.title {
    font-size: 40px;
    font-weight: 700;
    color: #2c7be5;
    text-align: center;
}

.subtitle {
    font-size: 18px;
    text-align: center;
    color: #555;
    margin-bottom: 30px;
}

.card {
    background-color: dark blue;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}

.stButton>button {
    background-color: #2c7be5;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 220px;
    font-size: 16px;
    font-weight: 600;
}

.stButton>button:hover {
    background-color: #1a5edb;
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

# ---------- Session Page ----------
if "page" not in st.session_state:
    st.session_state.page = "home"

# ---------- HOME ----------
if st.session_state.page == "home":

    st.markdown('<div class="title">🌿 AI Health Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Your personal AI health assistant</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
    <h3>💚 Welcome!</h3>
    <p>
    This smart AI tool analyzes your health data and predicts possible health risks.
    Click below to begin your assessment.
    </p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("➡️ Start Assessment"):
        st.session_state.page = "questionnaire"
        st.rerun()

# ---------- QUESTIONNAIRE ----------
elif st.session_state.page == "questionnaire":

    st.markdown('<div class="title">📝 Health Questionnaire</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)

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
        if st.button("⬅️ Back Home"):
            st.session_state.page = "home"
            st.rerun()

    with col2:
        if st.button("➡️ Predict"):
            st.session_state.page = "prediction"
            st.rerun()

# ---------- PREDICTION ----------
elif st.session_state.page == "prediction":

    st.markdown('<div class="title">🔍 Prediction Result</div>', unsafe_allow_html=True)

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
        <div class="card">
        <h2>🩺 Predicted Health Status: {disease}</h2>
        <p>This is an AI prediction and not a medical diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔄 Start Again"):
            st.session_state.page = "home"
            st.rerun()

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
