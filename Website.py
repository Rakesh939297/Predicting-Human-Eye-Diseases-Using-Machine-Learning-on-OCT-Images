import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile


# TensorFlow model prediction with confidence
def modelPrediction(testImagePath):
    model = tf.keras.models.load_model("MobilenetV3.keras")
    img = tf.keras.utils.load_img(testImagePath, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    predictions = model.predict(x)
    resultIndex = np.argmax(predictions)
    confidence = float(np.max(predictions))  # Convert to native Python float
    return resultIndex, confidence


# Set page config FIRST
st.set_page_config(page_title="Retinal OCT Analysis", layout="centered")

# Hide Streamlit style elements
hideStreamlitStyle = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""
st.markdown(hideStreamlitStyle, unsafe_allow_html=True)

# --- Main Disease Identification Page ---
st.header("Welcome to the Retinal OCT Analysis Platform")
st.markdown("Please upload a **retinal OCT image** only. The model is trained to detect: `CNV`, `DME`, `DRUSEN`, `NORMAL`.")

testImage = st.file_uploader("Upload your Image:")
if testImage is not None:
    # Save to a temporary file and get its path
    with tempfile.NamedTemporaryFile(delete=False, suffix=testImage.name) as tmpFile:
        tmpFile.write(testImage.read())
        testImagePath = tmpFile.name

# Predict button
if st.button("Predict") and testImage is not None:
    with st.spinner("Please Wait.."):
        resultIndex, confidence = modelPrediction(testImagePath)
        className = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
        predictedLabel = className[resultIndex]

    if confidence < 0.85:
        st.warning(f"⚠️ The model is unsure about this prediction.")
    else:
        st.success(f"✅ Model predicts as: **{predictedLabel}**")

        # Recommendation Section
        with st.expander("Learn More"):
            if resultIndex == 0:
                st.write("OCT scan showing *CNV with subretinal fluid.*")
            elif resultIndex == 1:
                st.write("OCT scan showing *DME with retinal thickening and intraretinal fluid.*")
            elif resultIndex == 2:
                st.write("OCT scan showing *drusen deposits in early AMD.*")
            elif resultIndex == 3:
                st.write("OCT scan showing a *normal retina with preserved foveal contour.*")
            st.image(testImage)
