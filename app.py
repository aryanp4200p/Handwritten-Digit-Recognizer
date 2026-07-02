"""
Streamlit web app: draw a digit, get a live CNN prediction.

Run locally:
    streamlit run app.py

Deploy for free on Streamlit Community Cloud (see README "Hosting" section).
"""

import numpy as np
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
from PIL import Image

st.set_page_config(page_title="Digit Recognizer", page_icon="✍️", layout="centered")


@st.cache_resource
def load_model():
    return keras.models.load_model("digit_recognizer_model.keras")


model = load_model()

st.title("✍️ Handwritten Digit Recognizer")
st.write("Draw a single digit (0-9) below, then click **Predict**.")

col1, col2 = st.columns([1, 1])

with col1:
    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=18,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.subheader("Prediction")
    predict_clicked = st.button("Predict", type="primary")
    result_placeholder = st.empty()
    chart_placeholder = st.empty()

if predict_clicked:
    if canvas_result.image_data is None or canvas_result.image_data.sum() == 0:
        st.warning("Please draw a digit first.")
    else:
        # Canvas gives RGBA; convert to grayscale, resize to 28x28 (MNIST format)
        img = Image.fromarray(canvas_result.image_data.astype("uint8")).convert("L")
        img = img.resize((28, 28), Image.LANCZOS)

        arr = np.array(img).astype("float32") / 255.0
        arr = arr.reshape(1, 28, 28, 1)

        probs = model.predict(arr, verbose=0)[0]
        predicted_digit = int(np.argmax(probs))
        confidence = float(np.max(probs)) * 100

        result_placeholder.markdown(
            f"### Predicted digit: **{predicted_digit}**\nConfidence: **{confidence:.1f}%**"
        )
        chart_placeholder.bar_chart(
            {"Probability": probs}, x_label="Digit", y_label="Probability"
        )

st.divider()
st.caption(
    "CNN trained on the MNIST dataset (~99% test accuracy). "
    "Built with TensorFlow/Keras + Streamlit."
)
