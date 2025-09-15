import streamlit as st
import numpy as np
import pickle
import os

# Load the trained model
with open("beer_model.pkl", "rb") as f:
    model = pickle.load(f)

# Title
st.title("🍺 Beer Servings Predictor")

# Image path relative to this script
image_path = os.path.join(os.path.dirname(__file__), "images", "beer.jpg")
st.image(image_path, use_column_width=True)  # use_column_width is safe for older Streamlit

# User inputs
beer = st.number_input("Beer Servings", min_value=0, max_value=500, value=50)
spirit = st.number_input("Spirit Servings", min_value=0, max_value=500, value=30)
wine = st.number_input("Wine Servings", min_value=0, max_value=500, value=20)

# Prediction
if st.button("Predict Total Alcohol (litres)"):
    features = np.array([[beer, spirit, wine]])
    pred = model.predict(features)
    st.success(f"Estimated Total Litres of Pure Alcohol: {pred[0]:.2f}")
