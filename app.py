import streamlit as st
import numpy as np
import cv2
from detect import model, processor, predict_image

st.title("🕵️ ViT Deepfake Detector")
st.subheader("Upload an image and see if it's real or fake")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    st.image(image, channels="BGR", caption="Uploaded Image")

    with st.spinner("Analyzing..."):
        label, confidence = predict_image(model, processor, image)

    st.markdown(f"**Result:** {label}")
    st.progress(int(confidence * 100))