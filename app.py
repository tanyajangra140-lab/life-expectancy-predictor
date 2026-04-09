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



X = df[['mortality','bmi','alcohol','schooling','sleep','exercise','stress','smoking']]
y = df['life_expectancy']

df = df.dropna()
df = df.astype(float)

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
sleep = st.slider("Sleep Hours", 0, 12)
exercise = st.slider("Exercise Days", 0, 7)
stress = st.slider("Stress Level", 1, 10)
smoking = st.selectbox("Smoking", ["No", "Yes"])

smoking_val = 1 if smoking == "Yes" else 0
alcohol = st.sidebar.slider("Alcohol Frequency", 0, 5)



# Show input summary
st.subheader("📋 Your Lifestyle Summary")
st.write(f"🛌 Sleep: {sleep} hrs | 🏃 Exercise: {exercise} days | 😰 Stress: {stress}/10")
st.write(df)
# Prediction
if st.button("Predict"):
    input_data = np.array([[mortality, bmi, alcohol, schooling, sleep, exercise, stress, smoking_val]])
    input_scaled = scaler.transform(input_data)

    result = model.predict(input_scaled)

    progress = int(result[0])
    st.progress(min(progress, 100))

    st.success(f"Predicted Life Expectancy: {int(final_result)} years")

    # Risk Level
    if result > 75:
        st.success("✅ Low Risk - Healthy Lifestyle")
    elif result > 65:
        st.warning("⚠️ Moderate Risk")
    else:
        st.error("❌ High Risk")

    # Suggestions
    st.subheader("💡 Suggestions to Improve")

    penalty = 0

if sleep < 5:
    penalty += 10
if exercise == 0:
    penalty += 10
if stress > 8:
    penalty += 10
if smoking_val == 1:
    penalty += 10

final_result = result[0] - penalty

# Footer
st.markdown("---")
st.caption("⚠️ This is an AI-based prediction and not medical advice.")
