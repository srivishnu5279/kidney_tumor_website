import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import os
import io

st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] {
            background-color: #a2a9f4;
        }

        /* Change sidebar background color */
        [data-testid="stSidebar"] {
            background-color: #000000;
        }

        h1 {
            text-align: center;
            color: #00000 !important;
            font-size: 2.5em;
        }

        /* Style buttons */
        .stButton>button {
            background-color: #1574e1 !important;
            color: white !important;
            border-radius: 8px;
            padding: 10px 20px;
            font-size: 1.2em;
        }

        
        .stButton>button:hover {
            background-color: #3e8e41 !important;
        }

        /* Center and style uploaded images */
        .uploaded-image {
            display: block;
            margin: auto;
            max-width: 80%;
            border-radius: 10px;
            border: 2px solid #ddd;
        }

        /* Stylish Prediction Box */
        .prediction-box {
            background-color: black;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            text-align: center;
            font-size: 1.5em;
            font-weight: bold;
            color: #333;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)



# Load models
binary_model = load_model("models/binary_model_kidney.h5")
multiclass_model = load_model("models/multiclass_model_kidney.h5")

def predict_pipeline(image):
    # Convert the PIL image to a BytesIO object
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")  # Save the image to bytes
    image_bytes.seek(0)  # Move to the start of the byte stream

    # Load the image using Keras' load_img function
    img = load_img(image_bytes, target_size=(224, 224))  
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    binary_pred = binary_model.predict(img_array)
    if binary_pred > 0.5:
        multiclass_pred = multiclass_model.predict(img_array)
        class_idx = np.argmax(multiclass_pred)
        class_labels = {0: 'Cyst', 1: 'Stone', 2: 'Tumor'}
        return f"Abnormal: {class_labels[class_idx]}"
    else:
        return "Normal"

# Streamlit UI
st.title("Kidney Tumor Prediction")

uploaded_file = st.file_uploader("Upload an X-ray image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width = 300)
    
    if st.button("Predict"):
        prediction = predict_pipeline(image)
        st.write(f"### Prediction: {prediction}")
