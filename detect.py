from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import numpy as np
import cv2

# Load model and processor once
model = ViTForImageClassification.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")
processor = ViTImageProcessor.from_pretrained("prithivMLmods/Deep-Fake-Detector-Model")

def predict_image(model, processor, img_bgr):
    # Convert BGR (OpenCV) to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Convert to PIL Image
    image = Image.fromarray(img_rgb)

    # Process image and run through model
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    label = model.config.id2label[predicted_class]
    confidence = torch.softmax(logits, dim=1)[0][predicted_class].item()
    return label, confidence

