import streamlit as st
import requests
from PIL import Image
import os

st.set_page_config(
    page_title="Plant Disease Predictor",
    layout="centered"
)

st.title("🌿 Plant Disease Prediction")

FLASK_URL = os.environ.get("FLASK_URL", "http://flask-api:5000")
PREDICT_ENDPOINT = FLASK_URL.rstrip("/") + "/predict"

uploaded_file = st.file_uploader(
    "Upload Plant Leaf Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    img = Image.open(uploaded_file)

    st.image(img, caption="Uploaded Image")

    if st.button("Predict"):

        files = {
            "file": (
                uploaded_file.name,
                uploaded_file.getvalue(),
                uploaded_file.type
            )
        }

        response = requests.post(PREDICT_ENDPOINT, files=files)

        if response.status_code == 200:
            result = response.json()

            st.success(
                f"Prediction: {result['predicted_class']}"
            )

            st.write(
                f"Confidence: {result['confidence']:.4f}"
            )

        else:
            st.error(response.text)