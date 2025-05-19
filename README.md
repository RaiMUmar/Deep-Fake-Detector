# 🕵️ Deep Fake Detector

Deep-Fake-Detector is a lightweight, real-time image-based deepfake detector built with a pretrained Vision Transformer (ViT) model and an intuitive Streamlit UI.

It uses the fine-tuned [prithivMLmods/Deep-Fake-Detector-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model) from Hugging Face to classify images as **"Real"** or **"Fake"**.

---

## ✨ Features

- ✅ Upload any `.jpg`, `.jpeg`, or `.png` image
- ✅ Detect AI-generated content in real time
- ✅ View result with confidence score and visual feedback
- ✅ Download a classification report
- ✅ Built with `Streamlit`, `PyTorch`, and `transformers`

---

## 🧠 Model

This app uses the pretrained model:
> **[`prithivMLmods/Deep-Fake-Detector-Model`](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model)**  
> Fine-tuned on a curated dataset of AI-generated and authentic human faces using ViT `base-patch16-224`.

---

## 🚀 Installation

1. **Clone the repository**

2. **Install dependencies**
```bash
pip install streamlit transformers torch torchvision opencv-python pillow
```

---

## 🖼️ Usage

Run the app using Streamlit:

```bash
python -m streamlit run app.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 📁 File Structure

```
Deep-Fake-Detector/
├── app.py               # Streamlit frontend
├── detect.py            # Image analysis + ViT model logic
└── README.md
```

---

## 🧩 To-Do / Future Improvements

- [ ] Add face detection pre-filter
- [ ] Add Grad-CAM saliency heatmap
- [ ] Extend to video frame-by-frame analysis
- [ ] Add REST API endpoint

---

## 🛡️ License

MIT License — free for educational, ethical, and research purposes.

---

## 🙌 Acknowledgements

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [ViT Architecture Paper (Google Research)](https://arxiv.org/abs/2010.11929)
- [prithivMLmods/Deep-Fake-Detector-Model](https://huggingface.co/prithivMLmods/Deep-Fake-Detector-Model)
