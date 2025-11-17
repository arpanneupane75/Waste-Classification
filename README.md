# 🌱 AI-Powered Waste Management System

## 🎯 Aim
To create an AI-powered solution that accurately identifies and classifies different types of waste using computer vision and deep learning, supporting better waste segregation and recycling practices.

---

## 🧠 Solution
This project implements an intelligent waste classification system capable of recognizing six waste categories:

**cardboard, glass, metal, paper, plastic, and trash**

The final and only model used in this system is a **fine-tuned InceptionV3** deep learning architecture.  
The web interface is built using **Streamlit**, and all predictions are handled through a **Flask backend API**.

---

## 🔄 Workflow

1. **Upload File** – Users upload an image (JPG, JPEG, PNG) or video (MP4).  
2. **Flask Backend** – File is sent to the backend for preprocessing and model inference.  
3. **Preprocessing** – Images are resized, normalized, and converted into model-compatible format.  
4. **Prediction** – The InceptionV3 model predicts the correct waste category.  
5. **Segregation Guidance** – The system suggests proper disposal methods for the predicted category.  

---

## ⭐ Key Features

- Accurate waste classification using InceptionV3  
- Real-time prediction for both images and videos  
- Grad-CAM heatmap visualization for explainability  
- Streamlit UI integrated with Flask backend  
- Guidance on proper waste disposal and recycling  
- Simple and clean architecture focused on a single optimized model  

---

## 🛠 Development Process

1. **Dataset Preparation**  
   Collected images for all six waste categories from public datasets and custom sources.  

2. **Preprocessing**  
   - Images resized to **384 × 512**  
   - Normalization and standardization  
   - Train–test split of **80:20**  

3. **Data Augmentation**  
   Applied horizontal flip, zoom, shift, and fill transformations to improve model robustness.

4. **Model Training (InceptionV3 Only)**  
   - Used transfer learning  
   - Unfrozen top layers for fine-tuning  
   - Softmax classification for 6 classes  
   - Adam optimizer with categorical crossentropy loss  

5. **Evaluation & Fine-Tuning**  
   Monitored accuracy, loss, and performance on the test set.

6. **Frontend + Backend Integration**  
   - Streamlit used for UI  
   - Flask used for inference API  
   - Includes Grad-CAM explanation support  

---

## 📊 Model Performance (InceptionV3)

| Metric               | Score |
|----------------------|--------|
| Training Accuracy    | 96%    |
| Validation Accuracy  | 87%    |
| Test Accuracy        | 87%    |

### Training Visualizations  
![Training VS validation Plot](./plot1.png)  


---

## 🚧 Challenges

- Confusion between visually similar materials (e.g., clear glass vs. plastic bottles)  
- Low-quality camera images affecting prediction  
- High GPU compute requirements for training InceptionV3  

---

## 📁 Project Structure  
📦 waste-classification  
 ┣ 📂 model  
 ┃ ┗ 📄 waste_inceptionv3.h5          # Trained InceptionV3 model  
 ┣ 📂 notebook  
 ┃ ┗ 📄 training.ipynb                # Model training & evaluation  
 ┣ 📂 src  
 ┃ ┣ 📄 gradcam.py                    # Grad-CAM implementation  
 ┃ ┣ 📄 shap_explain.py               # SHAP explainability script  
 ┃ ┗ 📄 preprocess.py                 # Image preprocessing utilities  
 ┣ 📂 streamlit_app  
 ┃ ┗ 📄 streamlit_app.py              # Main Streamlit UI  
 ┣ 📂 data  
 ┃ ┣ 📂 train                         # Training dataset  
 ┃ ┗ 📂 test                          # Testing dataset  
 ┣ 📂 images  
 ┃ ┗ 📄 sample.png                    # Sample prediction images  
 ┣ 📄 requirements.txt                 # Dependencies  
 ┣ 📄 README.md                        # Project documentation  
 ┗ 📄 .gitignore                       # Git ignored files
