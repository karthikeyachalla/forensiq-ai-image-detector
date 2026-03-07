# Product Requirements Document (PRD)
## AI vs Real Image Classification Detector

### 1. Project Overview
The **AI vs Real Image Detector** is a web-based forensic tool built to combat misinformation and deceptive synthetic media. It evaluates an uploaded image and determines whether it is a real photograph or AI-generated, returning a confidence score, detailed forensics, and feature heatmaps. 

### 2. Goals & Objectives
- **Detection:** Reliably classify an image as AI-generated or Real using a trained Convolutional Neural Network (MobileNetV2, ResNet50, EfficientNetB0).
- **Transparency:** Provide visual heatmaps (Grad-CAM) to explain *why* the model made its decision.
- **Forensics:** Supplement mathematical predictions with strict metadata and frequency analysis (EXIF, Error Level Analysis, FFT, OCR watermarks).
- **Usability:** Offer an ultra-modern, glassmorphism-styled user interface for immediate accessibility.

### 3. Target Audience
- **General Public:** Users looking to verify the authenticity of an image seen online.
- **Journalists & Fact-Checkers:** Professionals needing multi-layered forensic capabilities.
- **Researchers:** Deep-learning engineers analyzing AI artifacts.

### 4. Core Features
- **Drag-and-Drop Interface:** Accept `.jpg`, `.jpeg`, `.png`, and `.webp` images.
- **Multi-Model Support:** Toggle between highly optimized models (MobileNetV2, ResNet50, EfficientNetB0).
- **Metadata (EXIF) Scanning:** Check for software fingerprints (e.g., Photoshop) and absent camera signatures.
- **Forensic Pipeline:**
  - Fast Fourier Transform (FFT) analysis.
  - Error Level Analysis (ELA) for manipulation artifacts.
  - OCR Watermark Detection (e.g., DALL·E corner marks).
- **Explainable AI (XAI):** Integrated Grad-CAM++ to highlight suspicious or realistic regions on the original image.
- **Feedback Collection:** Users can submit feedback against predictions.

### 5. Future Roadmap
- Integration with Vision Transformers (ViT).
- Deploying Edge (mobile app) equivalents using quantized PyTorch arrays.
- Video stream analysis for deepfake detection.
