import os
# MacOS / TensorFlow fixes
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from PIL import Image
import numpy as np
import tensorflow as tf
from waitress import serve  # Production-safe WSGI server

# Limit TensorFlow threads to avoid mutex issues
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

app = Flask(__name__)

# Load your trained InceptionV3 model
model_path = "/Users/arpanneupane75/Downloads/Intern2/waste-classification/garbage_classification_model_inception.h5"
model = load_model(model_path)

# Define waste categories
waste_categories = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

# Preprocessing function
def preprocess_image(img: Image.Image):
    img = img.resize((384, 512))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    try:
        img = Image.open(file.stream).convert("RGB")
        processed = preprocess_image(img)

        prediction = model.predict(processed)
        predicted_index = int(np.argmax(prediction))
        predicted_category = waste_categories[predicted_index]
        probability = float(prediction[0][predicted_index])

        return jsonify({
            "prediction": predicted_category,
            "probability": probability
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run with Waitress for MacOS stability
if __name__ == "__main__":
    print("Starting Mac-safe Flask API with Waitress...")
    serve(app, host="0.0.0.0", port=5000)
