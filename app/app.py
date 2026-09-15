import streamlit as st
import numpy as np

# Load the model saved by Task 4
theta = np.load("results/setB_theta.npy")
mean = np.load("results/setB_mean.npy")
std = np.load("results/setB_std.npy")

st.title("Solar Power Prediction")

st.write("Enter public weather data to predict AC power.")

# User inputs
hour = st.number_input(
    "Hour of Day (0-23)",
    min_value=0,
    max_value=23,
    value=12
)

sw_radiation = st.number_input(
    "Shortwave Radiation (W/m²)",
    min_value=0.0,
    value=500.0
)

temp_2m = st.number_input(
    "Temperature at 2m (°C)",
    value=30.0
)

cloud_cover = st.number_input(
    "Cloud Cover (%)",
    min_value=0.0,
    max_value=100.0,
    value=20.0
)

if st.button("Predict AC Power"):

    # Convert hour to cyclic features
    sin_hour = np.sin(2 * np.pi * hour / 24)
    cos_hour = np.cos(2 * np.pi * hour / 24)

    # Same feature order used for Set B in Task 4
    x = np.array([
        sw_radiation,
        temp_2m,
        cloud_cover,
        sin_hour,
        cos_hour
    ])

    # Scale using Task 4 training statistics
    x_scaled = (x - mean) / std

    # Add intercept
    x_scaled = np.insert(x_scaled, 0, 1)

    # Predict
    prediction = x_scaled @ theta

    # Clip negative prediction to zero
    prediction = max(prediction, 0)

    st.success(f"Predicted AC Power: {prediction:.2f}")