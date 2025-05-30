
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import pickle

# Load your trained MNIST model

   model = load_model("image_classification.pkl") 

st.set_page_config(page_title="MNIST Digit Classifier", layout="centered")
st.title("🧠 MNIST Digit Classifier")
st.write("Draw a digit (0–9) below and let the model predict it.")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=10,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

if canvas_result.image_data is not None:
    img = canvas_result.image_data

    # Preprocess image: convert to 28x28 grayscale
    img = Image.fromarray((img[:, :, 0]).astype('uint8'))  # Take only 1 channel
    img = img.resize((28, 28)).convert('L')  # Resize and convert to grayscale
    img_arr = np.array(img) / 255.0          # Normalize
    img_arr = img_arr.reshape(1, 28, 28)  # Reshape for model input to match the training data shape

    if st.button("Predict"):
        pred = model.predict(img_arr)
        st.subheader(f"🧾 Prediction: {np.argmax(pred)}")
        st.bar_chart(pred[0])
