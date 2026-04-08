import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

data = {
    'age': [25, 40, 30, 50, 35, 60, 45, 28],
    'sleep_hours': [7, 5, 6, 4, 8, 5, 6, 7],
    'exercise_days': [5, 1, 3, 0, 4, 1, 2, 5],
    'smoking': [0, 1, 0, 1, 0, 1, 1, 0],
    'alcohol': [1, 3, 2, 4, 1, 3, 2, 1],
    'stress_level': [3, 8, 5, 9, 4, 7, 6, 3],
    'life_expectancy': [80, 65, 75, 60, 82, 62, 70, 78]
}

df = pd.DataFrame(data)

X = df.drop('life_expectancy', axis=1)
y = df['life_expectancy']

model = RandomForestRegressor()
model.fit(X, y)

st.title("🧠 Life Expectancy Predictor")

age = st.slider("Age", 10, 100)
sleep = st.slider("Sleep Hours", 0, 12)
exercise = st.slider("Exercise Days per Week", 0, 7)
smoking = st.selectbox("Smoking", [0, 1])
alcohol = st.slider("Alcohol Frequency", 0, 5)
stress = st.slider("Stress Level", 1, 10)

if st.button("Predict"):
    input_data = np.array([[age, sleep, exercise, smoking, alcohol, stress]])
    result = model.predict(input_data)

    st.success(f"Predicted Life Expectancy: {int(result[0])} years")
