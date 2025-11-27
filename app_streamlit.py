import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
from predict import predict_and_recommend

st.set_page_config(page_title="Skin Tone Classifier", layout="centered")

st.title("🎨 Skin Tone Classification & Makeup Recommender")
st.write("Upload a face image to predict your skin tone and get makeup suggestions.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show image
    st.image(uploaded_file, caption="Uploaded Image", width=300)

    # Save uploaded file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    st.write("⏳ Predicting...")
    tone, recs = predict_and_recommend(temp_path)

    # Display results
    st.subheader(f"🌟 Predicted Skin Tone: {tone.title()}")

    if recs:
        st.subheader("💄 Recommended Shades:")
        for category, items in recs.items():
            st.write(f"### {category.title()}")
            for item in items:
                st.write(f"- **{item['name']}** — `{item['hex']}`")
    else:
        st.error("No recommendations found for this tone.")
