
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("mnist.h5")

# Streamlit UI
st.title("🖊️ MNIST Handwritten Digit Recognition")
st.write("Upload an image of a handwritten digit (0-9)")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)  # Convert to NumPy array
    image = image / 255.0  # Normalize (0 to 1)
    image = 1 - image  # Invert colors (MNIST format)
    image = image.reshape(1, 28, 28, 1)  # Reshape for model input
    return image

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_digit = np.argmax(prediction)

    # Show result
    st.write(f"🖊️ Predicted Digit: **{predicted_digit}**")
