import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Health AI", page_icon="🌿", layout="wide")

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

# ---------- Navigation ----------
st.sidebar.title("🌿 Health AI")
page = st.sidebar.radio("Navigate", ["🏠 Home", "📝 Health Questionnaire", "🔍 Prediction"])

# ---------- HOME ----------
if page == "🏠 Home":
    st.title("🌿 AI Health Risk Predictor")
    st.write(
        """
        Welcome to your personal health assistant 💚  
        Answer a few simple questions and get your predicted health risk.
        """
    )

    st.image(
        "https://images.unsplash.com/photo-1498837167922-ddd27525d352",
        use_column_width=True
    )

# ---------- QUESTIONNAIRE ----------
elif page == "📝 Health Questionnaire":
    st.title("📝 Health Questionnaire")

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

    # Convert Yes/No to 0/1
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

    st.success("✅ Answers saved! Go to Prediction page")

# ---------- PREDICTION ----------
elif page == "🔍 Prediction":
    st.title("🔍 Prediction Result")

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

        st.success(f"🩺 Predicted Health Status: **{disease}**")

        st.info("💡 This is an AI prediction, not a medical diagnosis.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit")
