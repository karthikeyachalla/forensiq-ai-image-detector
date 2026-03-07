# Technical Design Document (TDD)
## AI vs Real Image Classification

### 1. System Architecture
This application follows a monolithic client-server architecture hosted purely in Python using **Streamlit**. 

- **Frontend:** Streamlit renders a highly customized HTML/CSS styling block on load, hijacking default UI components for a premium visual "glassmorphism" aesthetic.
- **Backend Analytics Engine:** Python execution block encompassing the deep-learning orchestrator (PyTorch execution) and image manipulation pipelines (OpenCV, PIL, Numpy).

### 2. Deep Learning Models
The application offers three interchangeable Convolutional Neural Networks (CNNs). All models were customized with a binary classification head and trained iteratively:
- **MobileNetV2** (Fast, Lightweight. Test Accuracy: 98.81%)
- **ResNet50** (Deep residual connections. Test Accuracy: 98.81%)
- **EfficientNetB0** (Compound scaled. Test Accuracy: 97.27%)

*Models are loaded in `app.py` directly from the `models/` directory using `@st.cache_resource` to avoid redundant memory allocation.*

### 3. Forensic Pipeline 

The inference goes through the following multi-agent evaluation pipeline before outputting a verdict:
1. **EXIF Scanner:** Uses `Pillow`'s `ExifTags` to check software fields versus hardware lenses.
2. **CNN Inference Classifier:** Passes properly transformed (224x224, Normalized) tensors to the active PyTorch model to yield a softmax probability.
3. **Grad-CAM++ (Gradient-weighted Class Activation Mapping):** Captures gradients from the final convolutional layer backward-passed from the target class label to project spatial activation points.
4. **Watermark OCR:** Converts the image to grayscale and uses EasyOCR (fallback simulated if library absent) to look for "DALL", "Midjourney", "Stable", etc.
5. **ELA (Error Level Analysis):** Re-compresses the image to `.jpg` at 90% quality and diffs the tensors to spot inconsistencies.
6. **FFT (Fast Fourier Transform):** Shifs spatial frequencies into spectral graphs to locate high-frequency repetition (common in Generative Adversarial Networks).

### 4. Data Flow
1. **User Upload:** File is streamed into memory bytes via `st.file_uploader`.
2. **Analysis Initiation:** System halts input and renders `st.status()` spinners.
3. **Sequential Execution:** The file runs sequentially through the Forensic Pipeline (above).
4. **Final Scoring:** The mathematical probabilities are aggregated into `.verdict-ai` or `.verdict-real` UI components.

### 5. Deployment Constraints
To host effectively on Streamlit Community Cloud:
- Dependencies are fixed strictly in `requirements.txt`.
- GPU calls (`.cuda()`) are conditionally managed (`torch.device("cpu")` fallback implemented directly in `app.py`).
- `.pth` binaries are excluded from Git to comply with generic 100MB limits; it is recommended to manage models externally via LFS or cloud storage if limits are exceeded.
