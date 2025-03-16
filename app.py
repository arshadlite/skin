import streamlit as st
import tensorflow as tf
import numpy as np
import urllib.request
import os
from tensorflow.keras.preprocessing import image

# Google Drive links (replace with your file IDs)
SKIN_TYPE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1edsi07yWCjO0E9JwR9sTZITfY8e3SrVG"
SKIN_DISEASE_MODEL_URL = "https://drive.google.com/uc?export=download&id=1Q47ojUfegyHNvj3Fh6mHAFyeQcEeYKiN"
# Download & Load Models
@st.cache_resource
def load_model_from_drive(url, filename):
    """Download model from Google Drive and load it"""
    if not os.path.exists(filename):
        st.write(f"Downloading {filename}... (This may take a while)")
        urllib.request.urlretrieve(url, filename)
    return tf.keras.models.load_model(filename)

# Load models
skin_type_model = load_model_from_drive(SKIN_TYPE_MODEL_URL, "skin_type_model.h5")
skin_disease_model = load_model_from_drive(SKIN_DISEASE_MODEL_URL, "skin_disease_model.h5")

# Define Class Labels
SKIN_TYPE_CLASSES = ["Dry", "Normal", "Oily"]
SKIN_DISEASE_CLASSES = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Melasma"]

# Skin Care Recommendations
def skincare_recommendations(skin_type, skin_disease):
    recommendations = {
        "Dry": "Use a heavy moisturizer, drink plenty of water, and avoid hot showers.",
        "Normal": "Maintain a balanced skincare routine with gentle cleansers and moisturizers.",
        "Oily": "Use oil-free skincare products, cleanse twice daily, and avoid heavy creams.",
        "Acne": "Use salicylic acid or benzoyl peroxide, avoid touching your face, and cleanse regularly.",
        "Eczema": "Use fragrance-free moisturizers, avoid harsh soaps, and stay hydrated.",
        "Psoriasis": "Use medicated creams, avoid triggers like stress, and keep skin moisturized.",
        "Rosacea": "Use gentle skincare, avoid spicy foods and extreme temperatures.",
        "Melasma": "Use sunscreen daily, avoid excessive sun exposure, and consider vitamin C serums."
    }
    return f"🛁 Skincare Tip: {recommendations.get(skin_type, '')} {recommendations.get(skin_disease, '')}"

# Image Preprocessing
def preprocess_image(image_file):
    img = image.load_img(image_file, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🔬 AI-Powered Skin Analysis")
st.write("Upload a skin image to detect **Skin Type** & **Skin Disease**, and get recommendations.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess Image
    img_array = preprocess_image(uploaded_file)
    
    # Predictions
    skin_type_pred = skin_type_model.predict(img_array)
    skin_disease_pred = skin_disease_model.predict(img_array)
    
    # Get Class Labels
    skin_type_result = SKIN_TYPE_CLASSES[np.argmax(skin_type_pred)]
    skin_disease_result = SKIN_DISEASE_CLASSES[np.argmax(skin_disease_pred)]
    
    # Display Results
    st.success(f"🩺 **Detected Skin Type:** {skin_type_result}")
    st.warning(f"⚕️ **Detected Skin Disease:** {skin_disease_result}")
    
    # Recommendations
    st.info(skincare_recommendations(skin_type_result, skin_disease_result))
