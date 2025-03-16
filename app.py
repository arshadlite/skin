import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array

# Load Model from Google Drive
@st.cache_resource
def load_model_from_drive(model_url):
    response = requests.get(model_url)
    if response.status_code == 200:
        model_path = model_url.split("/")[-1]  # Extract filename from URL
        with open(model_path, "wb") as f:
            f.write(response.content)
        return tf.keras.models.load_model(model_path)
    else:
        st.error("❌ Failed to download the model")
        return None

# Google Drive Model Links (Update with your actual links)
SKIN_TYPE_MODEL_URL = "https://drive.google.com/file/d/1edsi07yWCjO0E9JwR9sTZITfY8e3SrVG/view?usp=drive_link"
SKIN_DISEASE_MODEL_URL = "https://drive.google.com/file/d/1Q47ojUfegyHNvj3Fh6mHAFyeQcEeYKiN/view?usp=drive_link"

# Load the models
st.write("⏳ Loading Models... This may take a while.")
skin_type_model = load_model_from_drive(SKIN_TYPE_MODEL_URL)
skin_disease_model = load_model_from_drive(SKIN_DISEASE_MODEL_URL)

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.resize((150, 150))  # Resize to model input size
    img_array = img_to_array(image) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for prediction
    return img_array

# Function to predict skin type
def predict_skin_type(image):
    processed_image = preprocess_image(image)
    prediction = skin_type_model.predict(processed_image)
    categories = ["Dry", "Normal", "Oily"]
    return categories[np.argmax(prediction)]

# Function to predict skin disease
def predict_skin_disease(image):
    processed_image = preprocess_image(image)
    prediction = skin_disease_model.predict(processed_image)
    diseases = ["Acne", "Eczema", "Psoriasis", "Rosacea", "Melasma"]
    return diseases[np.argmax(prediction)]

# Streamlit UI
st.title("💆‍♂️ AI-Powered Skin Analysis")
st.write("Upload an image to detect **your skin type and possible skin disease.**")

uploaded_file = st.file_uploader("📂 Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_column_width=True)

    # Predict Skin Type
    skin_type = predict_skin_type(image)
    st.success(f"🧑‍⚕️ **Detected Skin Type:** {skin_type}")

    # Predict Skin Disease
    skin_disease = predict_skin_disease(image)
    st.warning(f"⚕️ **Possible Skin Disease:** {skin_disease}")

    # Recommendations
    st.subheader("💡 Personalized Skincare Recommendations")
    
    recommendations = {
        "Dry": "🔹 Use hydrating moisturizers. \n🔹 Avoid hot showers. \n🔹 Drink plenty of water.",
        "Oily": "🔹 Use oil-free skincare products. \n🔹 Wash face with gentle cleanser twice a day.",
        "Normal": "🔹 Maintain a balanced skincare routine. \n🔹 Apply sunscreen regularly.",
        "Acne": "🔹 Use salicylic acid cleansers. \n🔹 Avoid touching your face frequently.",
        "Eczema": "🔹 Apply fragrance-free moisturizers. \n🔹 Avoid allergens.",
        "Psoriasis": "🔹 Keep skin moisturized. \n🔹 Use medicated creams.",
        "Rosacea": "🔹 Avoid spicy food. \n🔹 Use gentle skincare products.",
        "Melasma": "🔹 Apply sunscreen daily. \n🔹 Use vitamin C serums."
    }
    
    st.info(f"💆‍♂️ **Skin Type Care:** {recommendations.get(skin_type, 'No recommendation available.')}")
    st.info(f"⚕️ **Disease Treatment:** {recommendations.get(skin_disease, 'No recommendation available.')}")
