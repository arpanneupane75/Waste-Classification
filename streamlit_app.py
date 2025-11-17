import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np

# -------------------------
# Load Model
# -------------------------
st.title("♻ Waste Classification App")
st.write("Upload an image to predict the waste category")

@st.cache_resource
def load_waste_model(model_path):
    return load_model(model_path)

model_path = "garbage_classification_model_inception.h5"
model = load_waste_model(model_path)

# Define categories
waste_categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# -------------------------
# Preprocessing function
# -------------------------
def preprocess_image(img: Image.Image):
    img = img.resize((384, 512))               # Match InceptionV3 input
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# -------------------------
# File uploader
# -------------------------
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", width=300)

    # Preprocess and predict
    processed_img = preprocess_image(img)
    prediction = model.predict(processed_img)
    predicted_index = int(np.argmax(prediction))
    predicted_category = waste_categories[predicted_index]
    probability = float(prediction[0][predicted_index])

    # Display results
    st.success(f"Prediction: **{predicted_category}**")
    st.info(f"Confidence: **{probability:.4f}**")
