# AI vs Real Image Classification Detector

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](#) *(Will link dynamically once deployed)*

## 📖 Project Overview
This project focuses on classifying images as **AI-generated** or **Real photographs** using deep learning models. It combines Deep Learning CNNs (MobileNetV2, ResNet50, EfficientNetB0) with Explainable AI (Grad-CAM++) and Forensic analysis pipelines (EXIF parsing, Error Level Analysis, FFT, and OCR Watermark Detection) to determine authenticity natively inside a web application.

---

## 📂 Repository Structure

```tree
ai-vs-real-image-classification/
│
├── docs/                      # Extensive engineering documentation
│   ├── PRD.md                 # Product Requirements Document
│   └── TDD.md                 # Technical Design Document
│
├── models/                    # .pth model binaries 
│   ├── efficientnet_balanced.pth
│   ├── mobilenet_balanced.pth
│   └── resnet50_balanced.pth  # Excluded via .gitignore for size 
│
├── notebooks/                 # Training environment Jupyter files
│   ├── EfficientNetB0.ipynb
│   ├── Main_model.ipynb
│   ├── MobilenetV2_Test.ipynb
│   └── ResNet50_Test.ipynb
│
├── app.py                     # Main execution file for the Streamlit dashboard
├── requirements.txt           # Dependency locking for Community Cloud integration
├── .gitignore                 # GitHub pipeline exclusion rules
└── README.md                  # Detailed execution guidelines
```

---

## 🎯 Objectives
- **Detection & Forensics:** Classify images into AI-generated and Real categories utilizing both Machine Learning arrays and traditional digital forensics.
- **Transparency:** Provide "Explainable AI" via Grad-CAM heatmaps overlaid directly on predictions to ensure visual traceability into the network. 
- **Compare:** Multiple CNN algorithms tested side-by-side using high-end optimization tuning on an extremely concise Test set (1,426 items).

---

## ⚙️ Technologies Used
- Feature Backend: **Python, PyTorch, Torchvision**
- Application Server (GUI): **Streamlit**
- Image Transformation: **NumPy, Matplotlib, OpenCV (python-headless), Scikit-learn, PIL**

---

## 🚀 How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/ai-vs-real-image-classification.git
cd ai-vs-real-image-classification
```

### 2️⃣ Start Virtual Environment & Install Dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3️⃣ Run the Dashboard Server
```bash
streamlit run app.py
```

*The application will launch directly into your browser via `http://localhost:8501`*

---

## ☁️ How to Deploy to Streamlit Community Cloud (for Public Web Access)

1. Push this repository sequentially to your public GitHub profile. Note: Ensure `models/*.pth` is handled (it's ignored by default if too large; use `Git LFS` to push `.pth` files dynamically overriding limit constraints).
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub account.
4. Click **New App**, select the `ai-vs-real-image-classification` repository.
5. Set the Main file path to `app.py`.
6. Click **Deploy!** 

Your app is now live with a public URL hosted securely for external users to analyze AI images without coding environments. 

---

## 👤 Author
Pothuri Indraneel  
B.Tech – Computer Science & Engineering  
