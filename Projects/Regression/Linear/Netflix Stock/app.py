import streamlit as st
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Netflix Stock Price Predictor",
    page_icon="📈",
    layout="centered"
)

# Load the model
@st.cache_resource
def load_model():
    file_path = "stock_price_model.pkl"
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['scaler']

try:
    model, scaler = load_model()
except FileNotFoundError:
    st.error("Model file 'stock_price_model.pkl' not found! Please place it in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Header
st.title("📈 Netflix Stock Price Predictor")
st.markdown("""
This app predicts the **Netflix Stock Price** based on expected market and calendar features.

The underlying model is trained on Netflix Stock Price data from **5th Feb 2018 to 5th Feb 2022**.
""")

# Input form
st.header("Enter Features")

col1, col2 = st.columns(2)

with col1:
    prev_close = st.number_input("Previous Close ($)", min_value=0.0, value=300.0, step=1.0)
    prev_return = st.number_input("Previous Return", value=0.001, step=0.001, format="%.5f")
    ma_7 = st.number_input("7-Day Moving Average ($)", min_value=0.0, value=295.0, step=1.0)
    ma_30 = st.number_input("30-Day Moving Average ($)", min_value=0.0, value=280.0, step=1.0)
    volatility = st.number_input("Volatility", min_value=0.0, value=0.02, step=0.005, format="%.4f")

with col2:
    log_volume = st.number_input("Volume", value=15.5, step=0.1)
    day_of_week = st.selectbox("Day of the Week", [0, 1, 2, 3, 4], format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][x])
    month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: pd.to_datetime(f"2023-{x}-01").strftime("%B"))
    quarter = st.selectbox("Quarter", [1, 2, 3, 4])

# Button to predict
if st.button("Predict Stock Price", type="primary"):
    # Create input DataFrame to match model expected feature names
    features = pd.DataFrame([{
        'Prev_Close': prev_close,
        'Prev_Return': prev_return,
        'MA_7': ma_7,
        'MA_30': ma_30,
        'Volatility': volatility,
        'Log_Volume': log_volume,
        'Day_of_Week': day_of_week,
        'Month': month,
        'Quarter': quarter
    }])
    
    # Scale features
    features_scaled = scaler.transform(features)

    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    # Display result
    st.success(f"### Predicted Stock Price: ${prediction[0]:.2f}")
    
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit. Model uses a Linear Regression approach on historical and synthetic data.")

# Run this app
# Navigate to the project directory and run "streamlit run app.py"

