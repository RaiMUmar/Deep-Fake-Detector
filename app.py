import streamlit as st
import numpy as np
import cv2
from io import StringIO
from detect import model, processor, predict_image

st.set_page_config(page_title="DeepScan AI", layout="centered")
st.title("🕵️ DeepScan AI")
st.subheader("Upload an image and detect if it's Real or AI-Generated")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        if image is None:
            raise ValueError("Could not decode image.")

        st.image(image, channels="BGR", caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(model, processor, image)

        # Confidence formatting
        confidence_percent = int(confidence * 100)
        st.markdown(f"### 🧠 Prediction: **{label}** ({confidence_percent}%)")

        # Visual cue based on prediction
        if label.lower() == "fake":
            st.error("⚠️ This image appears to be AI-generated (Deepfake).")
        elif label.lower() == "real":
            st.success("✅ This image appears authentic.")
        else:
            st.info("🤔 Unable to confidently classify this image.")

        # Confidence bar
        st.progress(confidence_percent)

        # Report download
        report = f"Prediction Report\n------------------\nResult: {label}\nConfidence: {confidence_percent}%"
        st.download_button("📄 Download Result Report", report, file_name="deepfake_report.txt")

    except Exception as e:
        st.error(f"❌ Error: {e}")
