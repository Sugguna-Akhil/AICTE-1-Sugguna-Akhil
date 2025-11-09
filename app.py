import streamlit as st
import pandas as pd
import joblib, os

st.set_page_config(page_title='EV Cost Predictor', layout='centered')
st.title('⚡ EV Cost Predictor')
st.write('Predict the price of an Electric Vehicle based on its specifications.')

MODEL_PATH = 'models/rf_ev_price.joblib'
SAMPLE = 'data/sample_ev_data.csv'

if os.path.exists(SAMPLE):
    df = pd.read_csv(SAMPLE)
    st.subheader('Sample Dataset Preview')
    st.dataframe(df.head())

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    st.subheader('Make a Prediction')
    battery = st.number_input('Battery Capacity (kWh)', value=50.0)
    power = st.number_input('Motor Power (kW)', value=150.0)
    range_km = st.number_input('Range (km)', value=300.0)
    brand = st.text_input('Brand', value='Generic')
    year = st.number_input('Model Year', value=2023)

    if st.button('Predict Price'):
        Xnew = pd.DataFrame([{
            'battery_kwh': battery,
            'power_kw': power,
            'range_km': range_km,
            'brand': brand,
            'year': year
        }])
        pred = model.predict(Xnew)[0]
        st.success(f'Predicted Price: ₹{pred:,.2f}')
else:
    st.warning('⚠️ Model not found. Please train it first using `python model/train_model.py`.')
