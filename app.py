import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="AI Career Recommendation",
    page_icon="🎯",
    layout="centered"
)

# Load trained model and encoders
model = pickle.load(open("model/career_model.pkl", "rb"))
interest_encoder = pickle.load(open("model/interest_encoder.pkl", "rb"))
career_encoder = pickle.load(open("model/career_encoder.pkl", "rb"))

st.title("🎯 AI Career Recommendation System")
st.markdown("### Find the best career based on your skills & interests")

st.sidebar.header("📌 About This Project")
st.sidebar.info(
    "This AI-powered system recommends suitable career paths "
    "using Machine Learning based on user skill assessment."
)

st.header("🔍 Enter Your Details")

aptitude = st.slider("Aptitude Score", 0, 100, 50)
programming = st.slider("Programming Skill", 0, 100, 50)
communication = st.slider("Communication Skill", 0, 100, 50)
logic = st.slider("Logical Thinking", 0, 100, 50)
interest = st.selectbox("Interest Area", ["tech", "management", "creative"])

if st.button("🎯 Predict Career"):
    interest_encoded = interest_encoder.transform([interest])[0]
    input_data = np.array([[aptitude, programming, communication, logic, interest_encoded]])

    probabilities = model.predict_proba(input_data)
    prediction_index = np.argmax(probabilities)
    confidence = np.max(probabilities) * 100

    career = career_encoder.inverse_transform([prediction_index])[0]

    st.success(f"🚀 **Recommended Career:** {career}")
    st.progress(int(confidence))
    st.write(f"📊 **Confidence:** {confidence:.2f}%")

st.markdown("---")
st.caption("⚠️ This system is for educational purposes only.")
