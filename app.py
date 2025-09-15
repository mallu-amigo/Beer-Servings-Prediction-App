import streamlit as st
import numpy as np
import pickle
import os


with open("beer_model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🍺 Beer Servings Predictor")
image_path = os.path.join(os.path.dirname(__file__), "images", "beer.jpg")
st.image(image_path, use_container_width=True)


beer = st.number_input("Beer Servings", min_value=0, max_value=500, value=50)
spirit = st.number_input("Spirit Servings", min_value=0, max_value=500, value=30)
wine = st.number_input("Wine Servings", min_value=0, max_value=500, value=20)

if st.button("Predict Total Alcohol (litres)"):
    pred = model.predict(np.array([[beer, spirit, wine]]))
    st.success(f"Estimated Total Litres of Pure Alcohol: {pred[0]:.2f}")


