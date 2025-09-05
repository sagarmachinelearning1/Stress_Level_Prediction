import streamlit as st

import streamlit as st
import numpy as np
import joblib
from src.predict import prediction



st.title("Stress Level Prediction App")

st.markdown("Fill out the details below to predict the **stress level**.")


age = st.number_input("Age", min_value=10, max_value=100, value=30)
coffee_intake = st.number_input("Coffee Intake (cups per day)", min_value=0.0, value=2.0)
caffeine_mg = st.number_input("Caffeine (mg)", min_value=0.0, value=200.0)
sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, value=7.0)
bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=24.0)
heart_rate = st.number_input("Heart Rate", min_value=40, max_value=200, value=75)
physical_activity = st.number_input("Physical Activity (hours per week)", min_value=0.0, value=5.0)
smoking = st.selectbox("Do you smoke?", [0, 1])
alcohol = st.selectbox("Do you consume alcohol?", [0, 1])


gender = st.selectbox("Gender", ["Male", "Female", "Other"])
gender_male = 1 if gender == "Male" else 0
gender_other = 1 if gender == "Other" else 0


countries = [
    'Belgium', 'Brazil', 'Canada', 'China', 'Finland', 'France', 'Germany',
    'India', 'Italy', 'Japan', 'Mexico', 'Netherlands', 'Norway', 'South Korea',
    'Spain', 'Sweden', 'Switzerland', 'UK', 'USA'
]
country = st.selectbox("Country", countries)
country_oh = [1 if c == country else 0 for c in countries]


sleep_quality = st.selectbox("Sleep Quality", ["Poor", "Fair", "Good"])
sq_poor = 1 if sleep_quality == "Poor" else 0
sq_fair = 1 if sleep_quality == "Fair" else 0
sq_good = 1 if sleep_quality == "Good" else 0


health_issues = st.selectbox("Health Issues", ["Unknow", "Moderate", "Severe"])
hi_moderate = 1 if health_issues == "Moderate" else 0
hi_severe = 1 if health_issues == "Severe" else 0
hi_unknow = 1 if health_issues == "Unknow" else 0


occupation = st.selectbox("Occupation", ["Office", "Service", "Student", "Other"])
occ_office = 1 if occupation == "Office" else 0
occ_service = 1 if occupation == "Service" else 0
occ_student = 1 if occupation == "Student" else 0
occ_other = 1 if occupation == "Other" else 0


input_data = np.array([[
    age, coffee_intake, caffeine_mg, sleep_hours, bmi, heart_rate,
    physical_activity, smoking, alcohol,
    gender_male, gender_other,
    *country_oh,
    sq_fair, sq_good, sq_poor,
    hi_moderate, hi_severe, hi_unknow,
    occ_office, occ_other, occ_service, occ_student
]])


if st.button("Predict Stress Level"):
    predicted_value = prediction(input_data)[0]
    stress_level = {
        0: "High Stress Level",
        1: "Low Stress Level",
        2: "Medium Stress Level"
    }

    # Get the corresponding label
    label = stress_level.get(predicted_value, "Unknown")

    st.success(f"Predicted Stress Level: **{label}**")
    
