import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Page config
st.set_page_config(page_title="Life Expectancy Predictor", page_icon="🧠", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🧠 Life Expectancy Predictor</h1>", unsafe_allow_html=True)
st.write("### Improve your lifestyle and see how it affects your life expectancy!")

# Sample dataset
df = pd.read_csv("life_expectancy.csv")



X = df.drop('life_expectancy', axis=1)
y = df['life_expectancy']

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train model
model = GradientBoostingRegressor()
model.fit(X, y)

model = RandomForestRegressor()
model.fit(X, y)

mortality = st.slider("Adult Mortality", 0, 500)
bmi = st.slider("BMI", 10, 40)
alcohol = st.slider("Alcohol Intake", 0, 10)
schooling = st.slider("Years of Schooling", 0, 20)


# Sidebar
st.sidebar.header("⚙️ Enter Your Details")

age = st.sidebar.slider("Age", 10, 100)
sleep = st.sidebar.slider("Sleep Hours", 0, 12)
exercise = st.sidebar.slider("Exercise Days/week", 0, 7)
smoking = st.sidebar.selectbox("Smoking", ["No", "Yes"])
alcohol = st.sidebar.slider("Alcohol Frequency", 0, 5)
stress = st.sidebar.slider("Stress Level", 1, 10)

# Convert categorical
smoking_val = 1 if smoking == "Yes" else 0

# Show input summary
st.subheader("📋 Your Lifestyle Summary")
st.write(f"🛌 Sleep: {sleep} hrs | 🏃 Exercise: {exercise} days | 😰 Stress: {stress}/10")

# Prediction
if st.button("Predict"):

    input_data = [[mortality, bmi, alcohol, schooling]]
    
    # Apply same scaling
    input_data = scaler.transform(input_data)

    result = model.predict(input_data)

    st.success(f"Predicted Life Expectancy: {int(result[0])} years")

    # Progress bar (visual effect)
    progress = int(result)
    st.progress(min(progress, 100))

    st.success(f"🎯 Expected Life Span: {int(result)} years")

    # Risk Level
    if result > 75:
        st.success("✅ Low Risk - Healthy Lifestyle")
    elif result > 65:
        st.warning("⚠️ Moderate Risk")
    else:
        st.error("❌ High Risk")

    # Suggestions
    st.subheader("💡 Suggestions to Improve")

    if sleep < 6:
        st.write("👉 Increase sleep to at least 7-8 hours")
    if exercise < 3:
        st.write("👉 Exercise at least 3-5 days per week")
    if stress > 7:
        st.write("👉 Practice meditation or relaxation techniques")
    if smoking_val == 1:
        st.write("👉 Avoid smoking for better health")
    if alcohol > 2:
        st.write("👉 Reduce alcohol consumption")

# Footer
st.markdown("---")
st.caption("⚠️ This is an AI-based prediction and not medical advice.")
