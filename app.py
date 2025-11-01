# app.py
import streamlit as st
import pandas as pd
import pickle
import openai
import os

# Set OpenAI API key (use your own key here or from environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load model
with open("model/ev_model.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="EV Cost Predictor", page_icon="⚡")

st.title("⚡ Electric Vehicle (EV) Cost Prediction with AI Insights")

st.markdown("Enter EV specifications below to predict the estimated cost.")

# User input
battery = st.number_input("🔋 Battery Capacity (kWh)", min_value=20.0, max_value=200.0, value=60.0)
power = st.number_input("⚙️ Motor Power (kW)", min_value=50.0, max_value=600.0, value=150.0)
range_km = st.number_input("📏 Range (km)", min_value=100.0, max_value=800.0, value=400.0)
weight = st.number_input("🚗 Vehicle Weight (kg)", min_value=800.0, max_value=3000.0, value=1500.0)
year = st.number_input("📅 Model Year", min_value=2015, max_value=2025, value=2024)

if st.button("🔮 Predict Price"):
    input_data = pd.DataFrame([[battery, power, range_km, weight, year]],
                              columns=["Battery_Capacity_kWh", "Motor_Power_kW", "Range_km", "Vehicle_Weight_kg", "Model_Year"])
    
    prediction = model.predict(input_data)[0]
    st.success(f"💰 Estimated Price: ₹{prediction:,.2f}")

    # Generative AI explanation
    prompt = f"""
    Explain why an electric vehicle with:
    - {battery} kWh battery,
    - {power} kW motor power,
    - {range_km} km range,
    - {weight} kg weight,
    - model year {year}
    would cost approximately ₹{prediction:,.2f}.
    Focus on how these features affect the cost.
    """

    with st.spinner("🤖 Generating AI explanation..."):
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert EV market analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        explanation = response["choices"][0]["message"]["content"]

    st.markdown("### 🧠 AI Explanation")
    st.write(explanation)
