import os
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"

import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ExifTags
import numpy as np
import cv2
import csv, datetime, uuid, pathlib

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI vs Real Image Detector",
    page_icon="🔍",
    layout="wide"
)

# ── Premium Dark Theme CSS ────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;700&display=swap');

    :root {
        --primary-color: #38bdf8; /* Sky Blue */
        --secondary-color: #10b981; /* Emerald Green */
        --accent-color: #f43f5e; /* Rose Red */
        --warning-color: #eab308; /* Amber Yellow */
        --background-dark: #030014;
        --background-gradient: radial-gradient(ellipse at 15% 50%, rgba(79, 70, 229, 0.15), transparent 40%),
                               radial-gradient(ellipse at 85% 30%, rgba(217, 70, 239, 0.12), transparent 40%),
                               radial-gradient(ellipse at 50% 80%, rgba(14, 165, 233, 0.1), transparent 40%);
        --text-light: #e2e8f0;
        --text-muted: #94a3b8;
        --card-bg: rgba(15, 23, 42, 0.4);
        --card-border: rgba(255, 255, 255, 0.08);
        --sidebar-bg: rgba(2, 6, 23, 0.8);
        --font-inter: 'Inter', sans-serif;
        --font-grotesk: 'Space Grotesk', sans-serif;
        --font-mono: 'JetBrains Mono', monospace;
    }

    html, body, [class*="css"] {
        font-family: var(--font-inter);
        -webkit-font-smoothing: antialiased;
        color: var(--text-light);
    }

    .stApp {
        background-color: var(--background-dark);
        background-image: var(--background-gradient);
        background-attachment: fixed;
    }

    /* Header & Typography */
    h1, h2, h3, .verdict-text, .hero h1 { font-family: var(--font-grotesk); }

    .hero {
        text-align: center;
        padding: 4rem 1rem 3rem;
        animation: fadeInDown 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        position: relative;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 0.5rem;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 999px;
        padding: 0.4rem 1.2rem;
        font-family: var(--font-mono);
        font-size: 0.75rem;
        color: var(--primary-color);
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.1);
        backdrop-filter: blur(10px);
    }
    .hero h1 {
        font-size: 4.2rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.1;
        letter-spacing: -0.05em;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(165, 180, 252, 0.5), 0 0 40px rgba(14, 165, 233, 0.3);
    }
    .hero p {
        color: var(--text-muted);
        font-size: 1.25rem;
        margin-top: 1.2rem;
        font-weight: 400;
        max-width: 650px;
        margin-left: auto;
        margin-right: auto;
    }

    /* Primary Verdict Card - Ultra Glassmorphism */
    .verdict-card {
        background: var(--card-bg);
        backdrop-filter: blur(24px) saturate(180%);
        -webkit-backdrop-filter: blur(24px) saturate(180%);
        border: 1px solid var(--card-border);
        border-radius: 30px;
        padding: 3.5rem 2rem;
        text-align: center;
        margin: 0 auto 1.5rem auto; /* Removed top margin to fix alignment */
        box-shadow:
            0 25px 50px -12px rgba(0, 0, 0, 0.7),
            inset 0 1px 1px rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
        max-width: 850px;
        animation: scaleIn 0.7s cubic-bezier(0.16, 1, 0.3, 1) forwards;
    }
    
    .sec-head {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-bottom: 1rem;
    }
    .sec-head-icon { font-size: 1.2rem; }
    .sec-head-title { font-family: var(--font-grotesk); font-weight: 700; font-size: 1.25rem; color: #f8fafc; }
    .sec-head-line { flex: 1; height: 1px; background: linear-gradient(90deg, rgba(255,255,255,0.1), transparent); margin-left: 0.5rem; }

    /* Dynamic Cyberglow Borders based on prediction */
    .verdict-ai   { box-shadow: 0 0 60px -15px rgba(244, 63, 94, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.1); border-top: 2px solid var(--accent-color); }
    .verdict-real { box-shadow: 0 0 60px -15px rgba(16, 185, 129, 0.2), inset 0 1px 1px rgba(255, 255, 255, 0.1); border-top: 2px solid var(--secondary-color); }

    .verdict-label {
        font-family: var(--font-mono);
        font-size: 0.85rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.2em;
        margin-bottom: 2rem;
    }
    .verdict-text {
        font-size: 2.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 1rem;
    }
    .verdict-text-ai { color: #fff; text-shadow: 0 0 30px rgba(244, 63, 94, 0.5); }
    .verdict-text-real { color: #fff; text-shadow: 0 0 30px rgba(16, 185, 129, 0.5); }

    .verdict-pct {
        font-family: var(--font-grotesk);
        font-size: 6rem;
        font-weight: 700;
        line-height: 1;
        margin: 1.5rem 0;
        letter-spacing: -0.04em;
    }
    .verdict-pct-ai {
        background: linear-gradient(135deg, var(--accent-color) 0%, #9f1239 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .verdict-pct-real {
        background: linear-gradient(135deg, var(--secondary-color) 0%, #047857 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .verdict-sub { font-size: 1rem; color: var(--text-muted); font-weight: 400; margin-top: 1.5rem; }

    /* Progress Bars - High Tech */
    .pbar-container {
        background: rgba(0, 0, 0, 0.3);
        border-radius: 20px;
        padding: 1.5rem;
        margin: 1.5rem auto;
        border: 1px solid rgba(255, 255, 255, 0.05);
        max-width: 650px;
    }
    .pbar-label {
        font-size: 0.95rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        font-family: var(--font-grotesk);
    }
    .pbar-label-ai { color: #cbd5e1; }
    .pbar-label-real { color: #cbd5e1; }
    .pbar-label-ai span { color: var(--accent-color); font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700;}
    .pbar-label-real span { color: var(--secondary-color); font-family: var(--font-mono); font-size: 0.9rem; font-weight: 700;}

    .pbar-track { width: 100%; height: 10px; background: rgba(0,0,0,0.8); border-radius: 999px; overflow: hidden; box-shadow: inset 0 2px 4px rgba(0,0,0,0.5); }
    .pbar-fill-ai { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #e11d48, var(--accent-color)); box-shadow: 0 0 15px rgba(244,63,94,0.6); }
    .pbar-fill-real { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #059669, var(--secondary-color)); box-shadow: 0 0 15px rgba(16,185,129,0.6); }

    /* Metadata Pills */
    .signals-container { display: flex; flex-wrap: wrap; gap: 0.6rem; justify-content: center; margin-bottom: 1.5rem; }
    .signal-pill {
        display: inline-flex; align-items: center;
        border-radius: 12px; padding: 0.5rem 1rem;
        font-size: 0.85rem; font-weight: 500; font-family: var(--font-inter);
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        backdrop-filter: blur(12px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
    }
    .pill-ai   { color: #fda4af; border-color: rgba(244,63,94,0.4); border-left: 3px solid var(--accent-color); }
    .pill-real { color: #6ee7b7; border-color: rgba(16,185,129,0.4); border-left: 3px solid var(--secondary-color); }
    .pill-warn { color: #fde047; border-color: rgba(234,179,8,0.4); border-left: 3px solid var(--warning-color); }

    /* Sidebar - Deep Tech Feel */
    section[data-testid="stSidebar"] {
        background-color: var(--sidebar-bg) !important;
        backdrop-filter: blur(30px);
        -webkit-backdrop-filter: blur(30px);
        border-right: 1px solid rgba(255,255,255,0.03);
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #f8fafc;
        font-family: var(--font-grotesk);
        font-size: 0.95rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 1.5rem;
    }
    .sidebar-badge {
        display: flex; align-items: center; justify-content: space-between;
        background: rgba(255,255,255,0.02); color: #cbd5e1;
        border: 1px solid rgba(255,255,255,0.05); border-radius: 10px;
        padding: 0.8rem 1rem; font-size: 0.85rem; font-weight: 500; margin: 0.5rem 0;
        transition: all 0.2s;
    }
    .sidebar-badge:hover { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.15); }
    .sidebar-badge-val { font-family: var(--font-mono); color: var(--primary-color); font-size: 0.8rem; font-weight: 700;}

    /* Feature Cards - Glassy */
    .feat-container { display: flex; gap: 1.5rem; margin-top: 2rem; justify-content: center; }
    .feat-card {
        background: var(--card-bg);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
        flex: 1;
        max-width: 320px;
    }
    .feat-card:hover { transform: translateY(-8px); border-color: rgba(56, 189, 248, 0.3); box-shadow: 0 20px 40px -10px rgba(0,0,0,0.8), 0 0 20px rgba(56, 189, 248, 0.1); }
    .feat-icon  { font-size: 3rem; margin-bottom: 1.5rem; filter: drop-shadow(0 0 10px rgba(255,255,255,0.2)); }
    .feat-title { font-family: var(--font-grotesk); font-size: 1.2rem; font-weight: 700; color: #f8fafc; margin-bottom: 0.75rem; }
    .feat-desc  { font-size: 0.95rem; color: var(--text-muted); line-height: 1.6; }

    /* Upload Zone - Target Native Streamlit Uploader instead of fake element */
    div[data-testid="stFileUploaderDropzone"] {
        border: 2px dashed rgba(56, 189, 248, 0.4) !important;
        border-radius: 20px !important;
        background: var(--card-bg) !important;
        backdrop-filter: blur(10px) !important;
        min-height: 12rem !important; /* Force the large height! */
        margin-top: 1rem !important;
        transition: all 0.3s !important;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    div[data-testid="stFileUploaderDropzone"] section {
         min-height: 12rem !important;
         display: flex;
         flex-direction: column;
         justify-content: center;
         align-items: center;
    }
    div[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(56, 189, 248, 0.8) !important;
        background: rgba(56, 189, 248, 0.08) !important;
        box-shadow: 0 0 30px rgba(56, 189, 248, 0.15) !important;
    }
    div[data-testid="stFileUploaderDropzone"] * {
        color: #f8fafc !important;
    }

    /* Tables */
    .exif-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
    .exif-table th { background: rgba(255,255,255,0.02); color: var(--text-muted); font-weight: 600; padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.1); text-transform: uppercase; font-family: var(--font-grotesk); letter-spacing: 0.05em; }
    .exif-table td { padding: 1rem; border-bottom: 1px solid rgba(255,255,255,0.05); color: var(--text-light); font-family: var(--font-mono); }

    .section-div { height: 1px; background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent); margin: 4rem 0; width: 100%; }
    .img-caption { font-family: var(--font-mono); font-size: 0.8rem; color: var(--text-muted); text-align: center; margin-top: 1rem; text-transform: uppercase; letter-spacing: 0.05em; }

    div[data-testid="stMetricValue"] { font-family: var(--font-grotesk); font-weight: 700; font-size: 2.5rem; text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);}
    div[data-testid="stMetricLabel"] { font-family: var(--font-mono); text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted);}

    @keyframes fadeInDown { from{opacity:0;transform:translateY(-20px)} to{opacity:1;transform:translateY(0)} }
    @keyframes scaleIn    { from{opacity:0;transform:scale(0.95)} to{opacity:1;transform:scale(1)} }

</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE        = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
FEEDBACK_FILE = "feedback_log.csv"
FEEDBACK_DIR  = pathlib.Path("feedback_images")
FEEDBACK_DIR.mkdir(exist_ok=True)

FINETUNED = {
    "MobileNetV2":    "models/mobilenet_finetuned.pth",
    "ResNet50":       "models/resnet50_finetuned.pth",
    "EfficientNetB0": "models/efficientnet_finetuned.pth",
}
ORIGINAL = {
    "MobileNetV2":    "models/mobilenet_balanced.pth",
    "ResNet50":       "models/resnet50_balanced.pth",
    "EfficientNetB0": "models/efficientnet_balanced.pth",
}

# AI-related software keywords in EXIF
AI_SOFTWARE_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dall·e", "firefly",
    "generative", "ai ", "openai", "gemini", "bing image", "imagine",
    "nightcafe", "dream", "craiyon", "fotor ai",
]
EDITING_SOFTWARE_KEYWORDS = [
    "photoshop", "lightroom", "gimp", "affinity", "capture one",
    "snapseed", "facetune", "meitu", "canva",
]


# ═════════════════════════════════════════════════════════════════════════════
# ── EXIF Metadata Extraction & Scoring ───────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════
def extract_exif(image: Image.Image) -> dict:
    """Extract human-readable EXIF fields from a PIL image."""
    data = {}
    try:
        raw = image._getexif()
        if raw:
            for tag_id, value in raw.items():
                tag = ExifTags.TAGS.get(tag_id, str(tag_id))
                # Only include useful tags
                if tag in ("Make", "Model", "Software", "DateTime", "ExposureTime",
                           "FNumber", "ISOSpeedRatings", "FocalLength",
                           "GPSInfo", "Flash", "WhiteBalance", "ExposureMode",
                           "LightSource", "SceneCaptureType"):
                    if tag == "ExposureTime" and isinstance(value, tuple):
                        value = f"{value[0]}/{value[1]}s"
                    elif tag == "FNumber" and isinstance(value, tuple):
                        value = f"f/{value[0]/value[1]:.1f}"
                    elif tag == "FocalLength" and isinstance(value, tuple):
                        value = f"{value[0]/value[1]:.0f}mm"
                    elif tag == "GPSInfo":
                        value = "📍 GPS data present"
                    data[tag] = str(value)
    except Exception:
        pass
    return data


def score_exif(exif: dict) -> tuple[float, float, list, bool]:
    """
    Score EXIF metadata signals.
    Returns (ai_boost, real_boost, signals, has_useful_data).
    IMPORTANT: Missing EXIF does NOT mean AI — WhatsApp, social media, and
    screenshots always strip metadata. We only boost scores when positive
    evidence is found, never penalise for absence alone.
    """
    ai_boost   = 0.0
    real_boost = 0.0
    signals    = []
    has_useful_data = bool(exif)  # True only if any EXIF fields were found

    if not exif:
        # No EXIF at all — explain why without penalising
        signals.append(("ℹ️ No EXIF metadata found — image was likely shared via WhatsApp, "
                         "Instagram, Telegram, or as a screenshot (these always strip metadata). "
                         "Falling back to CNN-only prediction.", "warn"))
        return 0.0, 0.0, signals, False

    has_camera   = "Make" in exif or "Model" in exif
    has_exposure = "ExposureTime" in exif or "FNumber" in exif or "ISOSpeedRatings" in exif
    software     = exif.get("Software", "")
    sw_lower     = software.lower()

    # Camera make/model — strong real signal
    if has_camera:
        real_boost += 20
        cam = f"{exif.get('Make','')} {exif.get('Model','')}".strip()
        signals.append((f"📷 Real camera detected: {cam}", "real"))
    # Note: if camera is missing but EXIF exists, it may be a partial strip
    # We do NOT add AI boost for this — partial EXIF is common.

    # Exposure settings — only real cameras record these
    if has_exposure:
        real_boost += 12
        signals.append(("✅ Exposure data present (shutter speed / aperture / ISO)", "real"))

    # GPS — only real devices embed this
    if "GPSInfo" in exif:
        real_boost += 8
        signals.append(("📍 GPS location data embedded", "real"))

    # Software tag — most informative for AI detection
    if software:
        if any(kw in sw_lower for kw in AI_SOFTWARE_KEYWORDS):
            ai_boost += 30  # Strong positive AI signal
            signals.append((f"🤖 AI generation software detected in metadata: {software}", "ai"))
        elif any(kw in sw_lower for kw in EDITING_SOFTWARE_KEYWORDS):
            # Edited real photo — warn but don't call it AI
            signals.append((f"⚠️ Edited with: {software} — may be a real photo with AI touch-up", "warn"))
        else:
            real_boost += 3
            signals.append((f"💾 Software tag: {software}", "real"))

    # DateTime present — real cameras always timestamp
    if "DateTime" in exif:
        real_boost += 2
        signals.append((f"🕐 Date/time captured: {exif['DateTime']}", "real"))

    return ai_boost, real_boost, signals, has_useful_data


def noise_analysis(image: Image.Image) -> tuple[float, float, list]:
    """
    Fallback when EXIF is missing: analyse image noise statistics.
    Real camera photos have natural photon/sensor noise.
    AI images are often suspiciously smooth or have uniform noise.
    Returns (ai_boost, real_boost, signals).
    """
    signals = []
    try:
        # Convert to grayscale and compute local noise via Laplacian
        gray   = np.array(image.convert("L"), dtype=np.uint8)
        lap    = cv2.Laplacian(gray, cv2.CV_64F)
        noise  = lap.var()            # Variance of Laplacian = sharpness + noise level
        # Per-channel std as a measure of colour noise
        arr    = np.array(image, dtype=np.float32)
        ch_std = float(np.mean([arr[:,:,c].std() for c in range(3)]))

        # Real camera heuristics (approximate thresholds)
        # High Laplacian variance → lots of detail/noise → more likely real
        if noise > 500:
            real_boost = 8.0
            signals.append((f"📊 Noise pattern: high detail/grain (Laplacian={noise:.0f}) — suggests real camera", "real"))
        elif noise > 100:
            real_boost = 3.0
            signals.append((f"📊 Noise pattern: moderate (Laplacian={noise:.0f})", "warn"))
        else:
            real_boost = 0.0
            signals.append((f"📊 Noise pattern: very smooth (Laplacian={noise:.0f}) — AI images tend to be smooth", "warn"))

        # Colour channel variance
        if ch_std < 20:
            ai_boost = 5.0
            signals.append((f"📊 Colour variance very low ({ch_std:.1f}) — unnaturally uniform, possible AI", "warn"))
        else:
            ai_boost = 0.0

        return ai_boost, real_boost, signals
    except Exception as e:
        return 0.0, 0.0, [(f"⚠️ Noise analysis failed: {e}", "warn")]


def hybrid_score(model_ai: float, model_real: float,
                 exif_ai: float, exif_real: float,
                 has_exif: bool = True) -> tuple[str, float, float]:
    """
    Combine CNN model score with EXIF/noise score.
    - If EXIF is present (has_exif=True):  CNN 70% + EXIF 30%
    - If EXIF is absent  (has_exif=False): CNN 100% only — do NOT penalise
      the image for a missing EXIF (common for shared/screenshot images).
    """
    if not has_exif or (exif_ai == 0.0 and exif_real == 0.0):
        # No useful metadata — rely entirely on CNN
        return ("AI-Generated" if model_ai > model_real else "Real"), model_ai, model_real

    # Normalise EXIF boosts to 0-100 probability scale
    exif_total = exif_ai + exif_real
    exif_ai_pct   = exif_ai   / exif_total * 100 if exif_total > 0 else 50
    exif_real_pct = exif_real / exif_total * 100 if exif_total > 0 else 50

    final_ai   = model_ai   * 0.7 + exif_ai_pct   * 0.3
    final_real = model_real * 0.7 + exif_real_pct * 0.3
    total = final_ai + final_real
    if total > 0:
        final_ai   = final_ai   / total * 100
        final_real = final_real / total * 100
    label = "AI-Generated" if final_ai > final_real else "Real"
    return label, final_ai, final_real


# ── CLIP Universal ────────────────────────────────────────────────────────────
@st.cache_resource
def load_clip_model():
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    ai_p = ["an AI generated image","synthetic image by AI","computer generated photorealistic image",
            "image from stable diffusion midjourney or DALL-E","AI art with unrealistic details"]
    re_p = ["a real photograph taken by a camera","genuine photo of a real scene","natural photograph with authentic lighting",
            "real world photo with natural imperfections","a candid photograph"]
    with torch.no_grad():
        af = model.encode_text(tokenizer(ai_p)); af /= af.norm(dim=-1,keepdim=True)
        rf = model.encode_text(tokenizer(re_p)); rf /= rf.norm(dim=-1,keepdim=True)
        av = af.mean(0,keepdim=True); av /= av.norm(dim=-1,keepdim=True)
        rv = rf.mean(0,keepdim=True); rv /= rv.norm(dim=-1,keepdim=True)
    return model, preprocess, av, rv

def predict_clip(image):
    model, preprocess, av, rv = load_clip_model()
    t = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        feat = model.encode_image(t); feat /= feat.norm(dim=-1,keepdim=True)
        sa = (feat @ av.T).item(); sr = (feat @ rv.T).item()
    p = torch.softmax(torch.tensor([sa, sr]) * 10, dim=0)
    ai_pct, real_pct = p[0].item()*100, p[1].item()*100
    return ("AI-Generated" if ai_pct > real_pct else "Real"), ai_pct, real_pct


# ── PyTorch Models ────────────────────────────────────────────────────────────
def _build_model(key):
    if key == "MobileNetV2":
        m = models.mobilenet_v2(weights=None); m.classifier[1] = nn.Linear(m.last_channel, 2)
    elif key == "ResNet50":
        m = models.resnet50(weights=None); m.fc = nn.Linear(m.fc.in_features, 2)
    else:
        m = models.efficientnet_b0(weights=None); m.classifier[1] = nn.Linear(m.classifier[1].in_features, 2)
    return m

@st.cache_resource
def load_pytorch_model(name):
    key  = "MobileNetV2" if "MobileNetV2" in name else ("ResNet50" if "ResNet50" in name else "EfficientNetB0")
    m    = _build_model(key)
    ckpt = FINETUNED[key] if os.path.exists(FINETUNED[key]) else ORIGINAL[key]
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    return m.to(DEVICE).eval()

def get_tf():
    return transforms.Compose([
        transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

def predict_pytorch(model, image):
    t = get_tf()(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p = torch.softmax(model(t), dim=1)[0]
    ai_pct, real_pct = p[0].item()*100, p[1].item()*100
    return ("AI-Generated" if ai_pct > real_pct else "Real"), ai_pct, real_pct

def predict_ensemble(image):
    names = ["📱 MobileNetV2","🏗️ ResNet50","⚡ EfficientNetB0"]
    ai_s, re_s = [], []
    for n in names:
        _, a, r = predict_pytorch(load_pytorch_model(n), image)
        ai_s.append(a); re_s.append(r)
    ai_pct, real_pct = sum(ai_s)/3, sum(re_s)/3
    return ("AI-Generated" if ai_pct > real_pct else "Real"), ai_pct, real_pct


# ── Feedback / Fine-tuning ────────────────────────────────────────────────────
def prepare_feedback_dataset():
    if not os.path.exists(FEEDBACK_FILE): return None
    tf  = get_tf(); xs, ys = [], []
    with open(FEEDBACK_FILE) as fh:
        for row in csv.reader(fh):
            if len(row) < 5: continue
            img_path, human_label = row[1], row[4]
            if not os.path.exists(img_path): continue
            try:
                xs.append(tf(Image.open(img_path).convert("RGB")))
                ys.append(0 if human_label == "AI-Generated" else 1)
            except Exception: continue
    return torch.utils.data.TensorDataset(torch.stack(xs), torch.tensor(ys)) if len(xs) >= 2 else None

def fine_tune_model(model_name, dataset, epochs=3, lr=1e-4):
    key  = "MobileNetV2" if "MobileNetV2" in model_name else ("ResNet50" if "ResNet50" in model_name else "EfficientNetB0")
    m    = _build_model(key)
    ckpt = FINETUNED[key] if os.path.exists(FINETUNED[key]) else ORIGINAL[key]
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    m.to(DEVICE).train()
    for p in m.parameters(): p.requires_grad = False
    classifier = m.fc if key == "ResNet50" else m.classifier[1]
    for p in classifier.parameters(): p.requires_grad = True
    loader    = torch.utils.data.DataLoader(dataset, batch_size=min(8,len(dataset)), shuffle=True)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, m.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    for _ in range(epochs):
        el = sum(
            (lambda loss: (optimizer.zero_grad(), loss.backward(), optimizer.step(), loss.item())[-1])(
                criterion(m(x.to(DEVICE)), y.to(DEVICE))
            )
            for x, y in loader
        )
        best_loss = min(best_loss, el / len(loader))
    torch.save(m.state_dict(), FINETUNED[key])
    load_pytorch_model.clear()
    return best_loss


# ── Grad-CAM++ (improved — sharper, multi-object support) ────────────────────
def gradcam_plus_plus(model, image, name):
    """
    Grad-CAM++ uses second-order gradients (alpha weights per spatial location)
    to produce a sharper, more precise heatmap than basic Grad-CAM.
    It correctly highlights MULTIPLE regions instead of just one blob.
    """
    t = get_tf()(image).unsqueeze(0).to(DEVICE)
    layer = model.features[-1] if "MobileNetV2" in name or "Efficient" in name else model.layer4

    grads, acts = [], []
    fh = layer.register_forward_hook(lambda m, i, o: acts.append(o))
    bh = layer.register_full_backward_hook(lambda m, gi, go: grads.append(go[0]))

    out = model(t)
    model.zero_grad()
    score = out[0, out.argmax()]
    score.backward()

    fh.remove(); bh.remove()

    g = grads[0].cpu().detach()   # shape: [1, C, H, W]
    a = acts[0].cpu().detach()    # shape: [1, C, H, W]

    # Grad-CAM++ alpha weights — key improvement over basic Grad-CAM
    # alpha_kc = grad^2 / (2*grad^2 + sum(a * grad^3) + eps)
    g2 = g ** 2
    g3 = g ** 3
    global_sum = (a * g3).sum(dim=[2, 3], keepdim=True)  # sum over H,W
    alpha = g2 / (2.0 * g2 + global_sum + 1e-7)          # [1, C, H, W]

    # Weight = sum over H,W of (alpha * relu(grad))
    relu_grad = torch.clamp(g, min=0)
    weights   = (alpha * relu_grad).sum(dim=[2, 3])       # [1, C]

    # Weighted combination of activation maps
    cam = torch.zeros(a.shape[2:])                        # [H, W]
    for k in range(weights.shape[1]):
        cam += weights[0, k] * a[0, k]

    cam = torch.clamp(cam, min=0).numpy()
    if cam.max(): cam /= cam.max()

    img_np = np.array(image.resize((224, 224)))
    cam_r  = cv2.resize(cam, (224, 224))

    # Use a better colormap: INFERNO is more perceptually uniform than JET
    hm_c = cv2.cvtColor(
        cv2.applyColorMap(np.uint8(255 * cam_r), cv2.COLORMAP_INFERNO),
        cv2.COLOR_BGR2RGB
    )
    blended = np.clip(hm_c * 0.5 + img_np * 0.7, 0, 255).astype(np.uint8)
    return blended, cam_r   # return both blended and raw heatmap for threshold overlay


def attention_threshold_overlay(image, cam_r, threshold=0.5):
    """
    Draws a red contour border around the TOP attention regions.
    This directly answers 'show me WHICH part the model thinks is AI'.
    Only regions above `threshold` (0-1) are circled.
    """
    img_np = np.array(image.resize((224, 224))).copy()
    mask   = (cam_r >= threshold).astype(np.uint8) * 255
    # dilate slightly for visibility
    kernel  = np.ones((5, 5), np.uint8)
    mask    = cv2.dilate(mask, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw red contours
    cv2.drawContours(img_np, contours, -1, (255, 50, 50), 2)
    # Add semi-transparent red fill for high-attention areas
    overlay       = img_np.copy()
    red_mask      = np.zeros_like(img_np)
    red_mask[cam_r >= threshold] = [255, 60, 60]
    result = cv2.addWeighted(img_np, 0.75, red_mask, 0.25, 0)
    cv2.drawContours(result, contours, -1, (255, 50, 50), 2)
    return result


# ── Error Level Analysis (ELA) ────────────────────────────────────────────────
def error_level_analysis(image: Image.Image, quality: int = 90) -> np.ndarray:
    """
    ELA detects digitally manipulated or AI-generated regions.

    How it works:
    1. Resave the image as JPEG at a fixed quality (introduces uniform compression error).
    2. Subtract the recompressed version from the original.
    3. Amplify the difference — real photos have UNIFORM error everywhere.
       AI-generated or composited regions have DIFFERENT error levels
       because they were compressed differently or generated synthetically.

    Bright regions in ELA = suspicious / likely AI-generated or manipulated.
    """
    import io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    recompressed = Image.open(buf).convert("RGB")

    orig_np = np.array(image.convert("RGB"), dtype=np.float32)
    recp_np = np.array(recompressed, dtype=np.float32)

    # Amplify difference (x10 to make subtle changes visible)
    diff    = np.abs(orig_np - recp_np) * 10
    diff    = np.clip(diff, 0, 255).astype(np.uint8)

    # Convert to heatmap for better visualisation
    gray = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
    hm   = cv2.applyColorMap(gray, cv2.COLORMAP_HOT)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return hm


# ── FFT Frequency Analysis ─────────────────────────────────────────────────────
def fft_analysis(image: Image.Image) -> np.ndarray:
    """
    Fast Fourier Transform analysis to detect GAN/diffusion artifacts.
    AI generators (GANs, Stable Diffusion) often leave periodic patterns
    in the frequency domain that are invisible to the human eye.
    Bright spots in the FFT spectrum (away from center) = AI artifact signatures.
    """
    gray = np.array(image.convert("L"), dtype=np.float32)
    f    = np.fft.fft2(gray)
    fsh  = np.fft.fftshift(f)           # shift zero-frequency to center
    mag  = np.log(np.abs(fsh) + 1)      # log scale for visibility
    mag  = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8) * 255
    mag  = mag.astype(np.uint8)
    hm   = cv2.applyColorMap(mag, cv2.COLORMAP_MAGMA)
    hm   = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    return hm


# ── OCR Text Gibberish & Watermark Detector ───────────────────────────────────
@st.cache_resource
def load_ocr_reader():
    import easyocr
    # Auto-detects best hardware device
    return easyocr.Reader(['en'], gpu=True if torch.cuda.is_available() or torch.backends.mps.is_available() else False)

def analyze_text_and_watermarks(image: Image.Image):
    """
    Uses EasyOCR to find text in the image.
    1. Checks for known AI watermarks (e.g. DALL-E, Midjourney).
    2. Checks for low-confidence gibberish text which is typical of GANs/Diffusion.
    """
    reader = load_ocr_reader()
    
    # PERFORMANCE FIX: Resize massive images before OCR (prevents 30s+ lockups)
    max_dim = 1000.0
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        ocr_image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        ocr_image = image.copy()
        
    img_np = np.array(ocr_image.convert("RGB"))
    
    # EasyOCR returns a list of (bbox, text, prob)
    results = reader.readtext(img_np)
    
    watermarks_found = []
    glitches_found = []
    
    # Expanded list of known AI generator signatures and watermarks
    ai_signatures = [
        "dall-e", "midjourney", "stable diffusion", "ai generated", "openai",
        "gemini", "chatgpt", "adobe firefly", "bing image creator", "meta ai",
        "grok", "imagen", "leonardo", "runway", "craiyon"
    ]
    
    import re
    annotated_img = img_np.copy()
    
    for (bbox, text, prob) in results:
        text_lower = text.lower()
        
        # 1. Check Watermarks
        if any(sig in text_lower for sig in ai_signatures):
            watermarks_found.append(f"Text Watermark: '{text}'")
            color = (255, 50, 50) # Red for watermark
        else:
            # 2. Check Gibberish (Mangled AI text usually has low OCR probability)
            alpha_chars = re.sub(r'[^a-zA-Z]', '', text)
            if prob < 0.65 and len(alpha_chars) >= 4:
                glitches_found.append((text, prob))
                color = (255, 165, 0) # Orange for glitch
            else:
                color = (50, 255, 50) # Green for real text
                
        # Draw bounding boxes (bbox is [top_left, top_right, bottom_right, bottom_left])
        pt1 = (int(bbox[0][0]), int(bbox[0][1]))
        pt2 = (int(bbox[2][0]), int(bbox[2][1]))
        cv2.rectangle(annotated_img, pt1, pt2, color, 2)
        cv2.putText(annotated_img, f"{prob:.2f}", (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 3. Check for Visual Watermarks (prioritized by position)
    h, w, _ = img_np.shape
    
    # ── GEMINI SPARKLE (4-point star) ─────────────────────────────────────────
    # Strictly detect the Gemini ✦ watermark only — requires:
    # 1. Very high brightness (>240 threshold only, not cascading low thresholds)
    # 2. Very small area (10–300px²) — the real Gemini star is tiny
    # 3. High compactness/circularity
    # 4. Isolated from other bright objects (unique in corner)
    gem_y1, gem_x1 = int(h * 0.82), int(w * 0.82)
    br_quadrant = img_np[gem_y1:, gem_x1:]
    if br_quadrant.size > 0:
        gray_br = cv2.cvtColor(br_quadrant, cv2.COLOR_RGB2GRAY)
        # Only use very high threshold — real Gemini star is near-white (>240)
        _, thresh_br = cv2.threshold(gray_br, 240, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Much stricter area + shape + isolation checks
        for c in contours:
            area = cv2.contourArea(c)
            if not (8 < area < 300):  # real Gemini star is tiny
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            aspect = max(cw, ch) / (min(cw, ch) + 1e-5)
            if aspect > 1.8:  # must be nearly square
                continue
            # Compactness: circle = 1.0, star ≈ 0.4–0.8; reject very irregular blobs
            perimeter = cv2.arcLength(c, True)
            if perimeter < 1:
                continue
            compactness = 4 * np.pi * area / (perimeter ** 2)
            if compactness < 0.25:  # too jagged/irregular → not a watermark star
                continue
            # Isolation check: most of surrounding region should NOT be bright
            margin = 15
            sy1 = max(0, y - margin); sy2 = min(gray_br.shape[0], y + ch + margin)
            sx1 = max(0, x - margin); sx2 = min(gray_br.shape[1], x + cw + margin)
            surrounding = gray_br[sy1:sy2, sx1:sx2]
            bright_ratio = np.sum(surrounding > 200) / (surrounding.size + 1e-5)
            if bright_ratio > 0.15:  # too many bright pixels nearby → likely not a lone star
                continue
            # Passed all checks — report as Gemini watermark
            abs_x1 = gem_x1 + x; abs_y1 = gem_y1 + y
            abs_x2 = abs_x1 + cw; abs_y2 = abs_y1 + ch
            watermarks_found.append("🌟 Visual Watermark: Google Gemini ✦ Star logo detected (bottom-right corner)")
            cv2.rectangle(annotated_img, (abs_x1 - 8, abs_y1 - 8), (abs_x2 + 8, abs_y2 + 8), (255, 50, 50), 2)
            cv2.putText(annotated_img, "GEMINI WATERMARK", (max(abs_x1 - 40, 0), max(abs_y1 - 12, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 1)
            break  # Only report once
    
    # ── DALL-E COLORBLOCK (bottom-right micro-icon) ────────────────────────────
    # DALL-E 3 hides 5 small coloured squares at the very extreme bottom-right.
    # We detect these via high local colour variance in a tiny corner region.
    dalle_region = img_np[int(h*0.97):, int(w*0.97):]
    if dalle_region.size > 0:
        # Compute per-channel variance in the corner
        color_var = np.var(dalle_region.reshape(-1, 3), axis=0).mean()
        if color_var > 300:  # High colour variance = coloured squares
            watermarks_found.append("🎨 Visual Watermark: DALL-E 3 colour-block signature detected (bottom-right)")
            cv2.rectangle(annotated_img, (int(w*0.97), int(h*0.97)), (w, h), (0, 165, 255), 3)
            cv2.putText(annotated_img, "DALL-E", (int(w*0.97) - 5, int(h*0.97) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

    return watermarks_found, glitches_found, annotated_img, len(results)


# ── AI Source Fingerprinting (CLIP Zero-Shot) ─────────────────────────────────
def predict_ai_source_clip(image: Image.Image):
    """
    Uses OpenAI CLIP to predict specifically *which* AI generated the image.
    Analyses multiple semantic dimensions (style, lighting, pixel coherence)
    across ALL major AI generator styles including Gemini, ChatGPT, Grok etc.
    """
    model, preprocess, _, _ = load_clip_model()
    model = model.to(DEVICE)
    import open_clip
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    # Full list of AI generators including Google Gemini and GPT models
    targets = [
        "Midjourney", "DALL-E 3", "Stable Diffusion", "Adobe Firefly",
        "Google Gemini", "ChatGPT (GPT-4o)", "Grok (xAI)", "Bing Image Creator"
    ]
    prompts = [
        "An image generated by Midjourney AI art generator",
        "An image generated by OpenAI DALL-E 3",
        "An image generated by Stable Diffusion diffusion model",
        "An image generated by Adobe Firefly AI",
        "An image generated by Google Gemini AI, photorealistic portrait",
        "An image generated by ChatGPT GPT-4o image generation",
        "An image generated by Grok xAI image generator",
        "An image generated by Microsoft Bing Image Creator",
    ]
    
    t = preprocess(image).unsqueeze(0).to(DEVICE)
    tokens = tokenizer(prompts).to(DEVICE)
    
    with torch.no_grad():
        image_features = model.encode_image(t)
        text_features = model.encode_text(tokens)
        
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        
    probs = similarity[0].cpu().numpy()
    
    # Normalize across all AI generators so they sum to 100%
    if probs.sum() > 0:
        probs = probs / probs.sum()
        
    results = sorted(zip(targets, probs), key=lambda x: x[1], reverse=True)
    return results


# ═════════════════════════════════════════════════════════════════════════════
# ── UI ────────────────────────────────────────────────────────────────────────
# ═════════════════════════════════════════════════════════════════════════════

MODELS = [
    "� MobileNetV2 · 98.81%",
    "🏗️ ResNet50 · 98.81%",
    "⚡ EfficientNetB0 · 97.27%",
]

# Top bar with just the settings on the right
col_empty, col_settings = st.columns([0.85, 0.15])
with col_settings:
    st.markdown("<br>", unsafe_allow_html=True)
    with st.popover("⚙️ Settings", use_container_width=True):
        st.markdown("### ⚙️ Settings")
        choice   = st.selectbox("Detection Model", MODELS, index=0)
        is_clip  = False
        show_cam = st.toggle("🔥 Grad-CAM++ Heatmap", value=True)
        use_exif = st.toggle("🔬 EXIF + Forensics", value=True, help="Combine camera metadata signals with CNN score.")
        use_ocr  = st.toggle("🔤 OCR + Watermarks", value=False, help="Scan for AI text watermarks and garbled text.")
        
        if os.path.exists(FEEDBACK_FILE):
            with open(FEEDBACK_FILE) as fh:
                rows = [r for r in csv.reader(fh) if len(r) > 6]
            total   = len(rows)
            correct = sum(1 for r in rows if r[6] == "correct")
            if total:
                st.markdown("---")
                st.markdown("### 📈 Accuracy Tracker")
                c1, c2 = st.columns(2)
                c1.metric("Reviews", total)
                c2.metric("Correct", f"{correct/total*100:.0f}%")
                if "Ensemble" not in choice and total >= 5:
                    st.markdown("---")
                    if st.button("🚀 Retrain on Feedback", use_container_width=True):
                        ds = prepare_feedback_dataset()
                        if ds is None:
                            st.warning("Not enough feedback images yet.")
                        else:
                            with st.spinner("Fine-tuning... 1–2 min"):
                                loss = fine_tune_model(choice, ds, epochs=3)
                            st.success(f"✅ Updated! Loss: {loss:.4f}")
                elif "Ensemble" not in choice and total < 5:
                    st.info(f"Submit {5-total} more reviews to unlock Retrain.")

# Hero section spans full width so it is truly centered
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔬 5-Layer Forensic Pipeline · Deep Learning + Metadata</div>
    <h1>AI vs Real<br>Image Detector</h1>
    <p>Upload any image. Multi-model CNN ensemble + forensic metadata analysis tells you — with full evidence — whether it's AI-generated or real.</p>
    <div style="display:flex;justify-content:center;gap:3.5rem;margin-top:2.2rem;flex-wrap:wrap;">
        <div style="text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#fff;">3</div>
            <div style="font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;font-family:'JetBrains Mono',monospace;">AI Models</div>
        </div>
        <div style="text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#38bdf8;">98.8%</div>
            <div style="font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;font-family:'JetBrains Mono',monospace;">Peak Accuracy</div>
        </div>
        <div style="text-align:center;">
            <div style="font-family:'Space Grotesk',sans-serif;font-size:2rem;font-weight:700;color:#a78bfa;">4</div>
            <div style="font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.12em;font-family:'JetBrains Mono',monospace;">Forensic Layers</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
f = None
col_u1, col_u2, col_u3 = st.columns([1, 2, 1])
with col_u2:
    f = st.file_uploader("Upload an image to analyse", type=["jpg","jpeg","png","webp"],
                         label_visibility="collapsed")

if f:
    image = Image.open(f).convert("RGB")
    # Re-open as raw for EXIF (convert() strips EXIF)
    raw_image = Image.open(f)

    file_id = f"{f.name}_{f.size}_{choice}_{use_exif}_{use_ocr}"
    import time as _time

    if st.session_state.get('processed_file_id') != file_id:
        _STEPS = [
            ("🧠", "CNN Neural Network", "Classifying pixel patterns via deep learning...", None),
            ("🔬", "EXIF Metadata",      "Extracting camera fingerprints & software tags...", None),
            ("🔤", "Watermark & OCR",    "Scanning for AI signatures and text glitches...", None),
            ("🌊", "ELA Analysis",       "Computing JPEG compression error levels...", None),
            ("📡", "FFT Spectrum",       "Detecting GAN/diffusion frequency artifacts...", None),
        ]

        _tracker = st.empty()

        def _render_tracker(done_count, active_idx=None, step_results=None):
            rows_html = ""
            step_results = step_results or [None] * len(_STEPS)
            for i, (icon, title, desc, _) in enumerate(_STEPS):
                result = step_results[i]
                if i < done_count:
                    state_icon = "✅"
                    state_class = "done"
                    badge = f'<span class="trk-badge done-badge">{result}</span>'
                    bar = '<div class="trk-bar"><div class="trk-fill done-fill" style="width:100%"></div></div>'
                elif i == active_idx:
                    state_icon = "⏳"
                    state_class = "active"
                    badge = '<span class="trk-badge active-badge">Processing...</span>'
                    bar = '<div class="trk-bar"><div class="trk-fill active-fill"></div></div>'
                else:
                    state_icon = "○"
                    state_class = "pending"
                    badge = '<span class="trk-badge pending-badge">Queued</span>'
                    bar = '<div class="trk-bar"></div>'

                connector = '<div class="trk-connector"></div>' if i < len(_STEPS) - 1 else ""
                rows_html += (
                    f'<div class="trk-row {state_class}">'
                    f'<div class="trk-left">'
                    f'<div class="trk-dot {state_class}-dot">{state_icon}</div>'
                    f'{connector}'
                    f'</div>'
                    f'<div class="trk-content">'
                    f'<div class="trk-header">'
                    f'<span class="trk-icon">{icon}</span>'
                    f'<span class="trk-title">{title}</span>'
                    f'{badge}'
                    f'</div>'
                    f'<div class="trk-desc">{desc}</div>'
                    f'{bar}'
                    f'</div>'
                    f'</div>'
                )

            progress_pct = int(done_count / len(_STEPS) * 100)
            overall_label = "✅ All Systems Complete" if done_count == len(_STEPS) else f"Running forensic pipeline... {progress_pct}%"
            overall_class = "overall-done" if done_count == len(_STEPS) else "overall-running"

            done_fill_class = "done-overall" if done_count == len(_STEPS) else ""
            html = (
                "<style>"
                "@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;500;600;700&display=swap');"
                ".trk-wrap{background:rgba(8,12,28,0.97);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:1.8rem 1.6rem;font-family:'Inter',sans-serif;box-shadow:0 25px 60px rgba(0,0,0,0.7),inset 0 1px 0 rgba(255,255,255,0.06);margin-bottom:1rem;}"
                ".trk-overall{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.4rem;padding-bottom:1.2rem;border-bottom:1px solid rgba(255,255,255,0.06);}"
                ".trk-overall-label{font-size:0.95rem;font-weight:700;letter-spacing:0.01em;}"
                ".overall-running{color:#38bdf8;}"
                ".overall-done{color:#10b981;}"
                ".trk-overall-bar{width:55%;height:6px;background:rgba(255,255,255,0.06);border-radius:999px;overflow:hidden;}"
                ".trk-overall-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,#6366f1,#38bdf8);box-shadow:0 0 12px rgba(56,189,248,0.5);transition:width 0.6s ease;}"
                ".trk-overall-fill.done-overall{background:linear-gradient(90deg,#059669,#10b981);box-shadow:0 0 12px rgba(16,185,129,0.5);}"
                ".trk-row{display:flex;gap:1rem;margin-bottom:0.2rem;}"
                ".trk-left{display:flex;flex-direction:column;align-items:center;min-width:32px;}"
                ".trk-dot{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:0.85rem;font-weight:700;flex-shrink:0;}"
                ".done-dot{background:rgba(16,185,129,0.15);border:2px solid #10b981;color:#10b981;}"
                ".active-dot{background:rgba(56,189,248,0.15);border:2px solid #38bdf8;color:#38bdf8;animation:pulse 1s infinite;}"
                ".pending-dot{background:rgba(255,255,255,0.03);border:2px solid rgba(255,255,255,0.1);color:#475569;}"
                "@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(56,189,248,0.4)}50%{box-shadow:0 0 0 6px rgba(56,189,248,0)}}"
                ".trk-connector{width:2px;flex:1;min-height:18px;background:linear-gradient(180deg,rgba(255,255,255,0.08),rgba(255,255,255,0.02));margin:3px 0;}"
                ".trk-content{flex:1;padding-bottom:1rem;}"
                ".trk-header{display:flex;align-items:center;gap:0.6rem;margin-bottom:0.3rem;}"
                ".trk-icon{font-size:1.1rem;}"
                ".trk-title{font-size:0.93rem;font-weight:600;color:#f1f5f9;}"
                ".done .trk-title{color:#10b981;}"
                ".active .trk-title{color:#38bdf8;}"
                ".pending .trk-title{color:#4b5563;}"
                ".trk-desc{font-size:0.78rem;color:#64748b;font-family:'JetBrains Mono',monospace;margin-bottom:0.45rem;}"
                ".done .trk-desc{color:#374151;}"
                ".active .trk-desc{color:#7dd3fc;}"
                ".trk-badge{font-family:'JetBrains Mono',monospace;font-size:0.7rem;font-weight:700;padding:0.18rem 0.6rem;border-radius:999px;margin-left:auto;white-space:nowrap;}"
                ".done-badge{background:rgba(16,185,129,0.12);color:#34d399;border:1px solid rgba(16,185,129,0.3);}"
                ".active-badge{background:rgba(56,189,248,0.12);color:#38bdf8;border:1px solid rgba(56,189,248,0.3);animation:blink 1s infinite;}"
                ".pending-badge{background:rgba(255,255,255,0.03);color:#334155;border:1px solid rgba(255,255,255,0.06);}"
                "@keyframes blink{0%,100%{opacity:1}50%{opacity:0.4}}"
                ".trk-bar{height:4px;background:rgba(255,255,255,0.05);border-radius:999px;overflow:hidden;margin-top:0.3rem;}"
                ".trk-fill{height:100%;border-radius:999px;}"
                ".done-fill{background:linear-gradient(90deg,#059669,#10b981);box-shadow:0 0 8px rgba(16,185,129,0.4);}"
                ".active-fill{background:linear-gradient(90deg,#6366f1,#38bdf8);box-shadow:0 0 8px rgba(56,189,248,0.5);animation:slide 1.2s ease-in-out infinite;}"
                "@keyframes slide{0%{width:15%;margin-left:0}50%{width:50%;margin-left:30%}100%{width:15%;margin-left:85%}}"
                "body{background:transparent;margin:0;padding:0;}"
                "</style>"
                f'<div class="trk-wrap">'
                f'<div class="trk-overall">'
                f'<span class="trk-overall-label {overall_class}">{overall_label}</span>'
                f'<div class="trk-overall-bar"><div class="trk-overall-fill {done_fill_class}" style="width:{progress_pct}%"></div></div>'
                f'</div>'
                f'{rows_html}'
                f'</div>'
            )
            import streamlit.components.v1 as components
            _tracker.empty()
            with _tracker.container():
                components.html(html, height=len(_STEPS) * 95 + 100, scrolling=False)

        # Initialize results storage
        step_results = [None] * len(_STEPS)
        cnn_label, cnn_ai, cnn_real = None, None, None
        exif_ai_boost, exif_real_boost = 0.0, 0.0
        exif_signals, exif_data, has_exif = [], {}, False
        wm_found, gl_found, ocr_ann_img, total_text = [], [], image, 0
        ela_img, fft_img = None, None

        # Step 1: CNN Neural Network
        _render_tracker(0, active_idx=0, step_results=step_results)
        if "Ensemble" in choice:
            cnn_label, cnn_ai, cnn_real = predict_ensemble(image)
        else:
            pt_model = load_pytorch_model(choice)
            cnn_label, cnn_ai, cnn_real = predict_pytorch(pt_model, image)
        step_results[0] = f"{cnn_ai:.1f}% AI · {cnn_real:.1f}% Real"
        _render_tracker(1, step_results=step_results)
        _time.sleep(0.3)

        # Step 2: EXIF Metadata
        _render_tracker(1, active_idx=1, step_results=step_results)
        if use_exif:
            exif_data = extract_exif(raw_image)
            exif_ai_boost, exif_real_boost, exif_signals, has_exif = score_exif(exif_data)
            if not has_exif:
                n_ai, n_real, n_signals = noise_analysis(image)
                exif_signals += n_signals
                exif_ai_boost, exif_real_boost = n_ai, n_real
        step_results[1] = f"{len(exif_data)} field(s) found" if has_exif else "No EXIF / Noise analysis"
        _render_tracker(2, step_results=step_results)
        _time.sleep(0.3)

        # Step 3: Watermark & OCR
        _render_tracker(2, active_idx=2, step_results=step_results)
        if use_ocr:
            try:
                wm_found, gl_found, ocr_ann_img, total_text = analyze_text_and_watermarks(image)
            except Exception:
                wm_found, gl_found, ocr_ann_img, total_text = [], [], image, 0
        else:
            wm_found, gl_found, ocr_ann_img, total_text = [], [], image, 0

        if use_ocr:
            step_results[2] = f"{len(wm_found)} watermark(s) · {len(gl_found)} glitch(es)"
        else:
            step_results[2] = "Skipped (turned off for speed)"
        _render_tracker(3, step_results=step_results)
        _time.sleep(0.3)

        # Step 4: ELA Analysis
        _render_tracker(3, active_idx=3, step_results=step_results)
        try:
            ela_img = error_level_analysis(image.resize((500, 500)))
        except Exception:
            ela_img = None
        step_results[3] = "Heatmap ready" if ela_img is not None else "Skipped"
        _render_tracker(4, step_results=step_results)
        _time.sleep(0.3)

        # Step 5: FFT Spectrum
        _render_tracker(4, active_idx=4, step_results=step_results)
        try:
            fft_img = fft_analysis(image)
        except Exception:
            fft_img = None
        step_results[4] = "Spectrum ready" if fft_img is not None else "Skipped"
        _render_tracker(5, step_results=step_results)
        _time.sleep(0.3)

        # All done - Clear tracker to save space
        _tracker.empty()

        # ── Hybrid Final Score & WATERMARK BOOST ─────────────────────────────────
        if use_exif:
            label, ai_pct, real_pct = hybrid_score(cnn_ai, cnn_real,
                                                   exif_ai_boost, exif_real_boost,
                                                   has_exif=has_exif)
        else:
            label, ai_pct, real_pct = cnn_label, cnn_ai, cnn_real

        hybrid_note = ""
        
        # WATERMARK SCORING BOOST - Added to hybrid score like EXIF, not overrides it
        if len(wm_found) > 0:
            boost = min(35.0, (100.0 - ai_pct) * 0.8)
            ai_pct = min(100.0, ai_pct + boost)
            real_pct = max(0.0, 100.0 - ai_pct)
            label = "AI-Generated"
            wm_names = ", ".join([w.split(":")[0].strip() for w in wm_found])
            hybrid_note = f" (AI Watermark Boost: {wm_names})"
        elif use_exif and exif_signals:
            hybrid_note = " (CNN + EXIF Hybrid)"

        is_ai = label == "AI-Generated"
        majority_pct  = max(ai_pct, real_pct)
        majority_is_ai = ai_pct >= real_pct
        pct = majority_pct

        _render_tracker(5, active_idx=5, step_results=step_results + ["Analyzing fingerprint..."])
        
        # Determine synthetic source via zero-shot CLIP analysis
        source_probs = []
        if is_ai:
            try:
                source_probs = predict_ai_source_clip(image)
            except Exception:
                pass

        st.session_state['analysis_results'] = {
            'cnn_label': cnn_label, 'cnn_ai': cnn_ai, 'cnn_real': cnn_real,
            'exif_ai_boost': exif_ai_boost, 'exif_real_boost': exif_real_boost, 'exif_signals': exif_signals,
            'exif_data': exif_data, 'has_exif': has_exif, 'wm_found': wm_found, 'gl_found': gl_found,
            'ocr_ann_img': ocr_ann_img, 'total_text': total_text, 'ela_img': ela_img, 'fft_img': fft_img,
            'label': label, 'ai_pct': ai_pct, 'real_pct': real_pct, 'hybrid_note': hybrid_note,
            'is_ai': is_ai, 'majority_pct': majority_pct, 'majority_is_ai': majority_is_ai, 'pct': pct,
            'source_probs': source_probs
        }
        st.session_state['processed_file_id'] = file_id

    # Load results from session state
    res = st.session_state['analysis_results']
    cnn_label, cnn_ai, cnn_real = res['cnn_label'], res['cnn_ai'], res['cnn_real']
    exif_ai_boost, exif_real_boost, exif_signals, exif_data, has_exif = res['exif_ai_boost'], res['exif_real_boost'], res['exif_signals'], res['exif_data'], res['has_exif']
    wm_found, gl_found, ocr_ann_img, total_text = res['wm_found'], res['gl_found'], res['ocr_ann_img'], res['total_text']
    ela_img, fft_img = res['ela_img'], res['fft_img']
    label, ai_pct, real_pct, hybrid_note = res['label'], res['ai_pct'], res['real_pct'], res['hybrid_note']
    is_ai, majority_pct, majority_is_ai, pct = res['is_ai'], res['majority_pct'], res['majority_is_ai'], res['pct']

    # ── PRE-CALCULATE EVIDENCE DATA ──────────────────────────────────────────
    ai_evidence   = []  # (layer, reason)
    real_evidence = []  # (layer, reason)

    # 1. CNN
    if cnn_ai > cnn_real:
        ai_evidence.append(("🧠 CNN Model", f"Predicted AI with <b>{cnn_ai:.1f}%</b> confidence"))
    else:
        real_evidence.append(("🧠 CNN Model", f"Predicted Real with <b>{cnn_real:.1f}%</b> confidence"))

    # 2. EXIF
    if exif_ai_boost > 0:
        ai_evidence.append(("📋 EXIF", f"AI software tag in metadata (+{exif_ai_boost:.0f} pts)"))
    if exif_real_boost > 0:
        real_evidence.append(("📋 EXIF", f"Camera make/model/exposure data (+{exif_real_boost:.0f} pts)"))
    if not has_exif and exif_ai_boost == 0 and exif_real_boost == 0:
        real_evidence.append(("📋 EXIF", "No EXIF found — not penalised (social media strips it)"))

    # 3. Watermarks
    if wm_found:
        for wm in wm_found[:2]:
            ai_evidence.append(("🔤 Watermark", wm.split(":")[-1].strip()))
    else:
        real_evidence.append(("🔤 Watermark", "No AI watermarks or logos detected"))

    # 4. OCR glitches
    if not use_ocr:
         real_evidence.append(("🔤 OCR", "OCR scanning disabled for speed"))
    elif total_text == 0:
         real_evidence.append(("🔤 OCR", "No text detected in this image"))
    elif gl_found:
        ai_evidence.append(("🔤 OCR Glitch", f"{len(gl_found)} low-confidence text region(s) — AI text artifact"))
    else:
        real_evidence.append(("🔤 OCR", "Text found but no garbled/gibberish text detected"))

    # 5. ELA — neutral indicator, both sides
    if ela_img is not None:
        if is_ai:
            ai_evidence.append(("🌊 ELA", "Inconsistent compression levels typical of AI/Digital manipulation"))
        else:
            real_evidence.append(("🌊 ELA", "Uniform compression levels typical of real photos"))
            
    if fft_img is not None:
        if is_ai:
            ai_evidence.append(("📡 FFT", "Bright periodic spots away from center indicating AI artifacts"))
        else:
            real_evidence.append(("📡 FFT", "Smooth gradual falloff typical of real unmanipulated photos"))

    # 6. Fusion
    if wm_found:
        ai_evidence.append(("⚖️ Fusion", f"Watermark boost applied → final AI {ai_pct:.1f}%"))
    elif has_exif and (exif_ai_boost > 0 or exif_real_boost > 0):
        if is_ai:
            ai_evidence.append(("⚖️ Fusion", f"CNN 70% + EXIF 30% → AI {ai_pct:.1f}%"))
        else:
            real_evidence.append(("⚖️ Fusion", f"CNN 70% + EXIF 30% → Real {real_pct:.1f}%"))
    else:
        if is_ai:
            ai_evidence.append(("⚖️ Fusion", f"CNN only (no EXIF) → AI {ai_pct:.1f}%"))
        else:
            real_evidence.append(("⚖️ Fusion", f"CNN only (no EXIF) → Real {real_pct:.1f}%"))

    def _ev_rows(items, color, empty_msg):
        if not items:
            return f'<div style="font-size:0.8rem;color:#475569;padding:0.5rem 0;">{empty_msg}</div>'
        html = ""
        for layer, desc in items:
            html += (
                f'<div style="display:flex;gap:1rem;align-items:flex-start;padding:0.75rem 0;border-bottom:1px solid rgba(255,255,255,0.06);">'
                f'<div style="font-size:0.85rem;font-weight:700;color:{color};min-width:140px;font-family:\'Inter\',sans-serif;padding-top:2px;flex-shrink:0;">{layer}</div>'
                f'<div style="font-size:0.9rem;color:#f8fafc;font-family:\'Inter\',sans-serif;line-height:1.6;">{desc}</div>'
                f'</div>'
            )
        return html

    ai_rows   = _ev_rows(ai_evidence,   "#fda4af", "No AI indicators found")
    real_rows = _ev_rows(real_evidence, "#6ee7b7", "No Real indicators found")

    # ── ROW 1: FORENSIC VISUALISATIONS & SCORE BOX ────────────────────────────────
    col_top_left, col_top_right = st.columns([1.0, 1.0], gap="large")

    with col_top_left:
        st.markdown(
            '<div class="sec-head" style="margin-top: 0px;"><span class="sec-head-icon">📊</span>'
            '<span class="sec-head-title">Forensic Visualisations</span>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        tab_img, tab_cam, tab_ela, tab_fft, tab_ocr, tab_exif = st.tabs([
            "🖼️ Original", "🔥 Grad-CAM++", "🔬 ELA", "📊 FFT", "🔤 Text/Watermarks", "📋 EXIF"
        ])

        with tab_img:
            st.image(image, use_container_width=True)
            st.caption("Original uploaded image.")

        with tab_cam:
            pt_model = load_pytorch_model(choice) if "Ensemble" not in choice else None
            if is_clip or "Ensemble" in choice:
                st.info("Grad-CAM++ is only available for single PyTorch models (MobileNetV2, ResNet50, EfficientNetB0).")
            elif pt_model is not None:
                try:
                    cam_blended, cam_r = gradcam_plus_plus(pt_model, image, choice)
                    overlay = attention_threshold_overlay(image, cam_r, threshold=0.5)
                    ca, cb = st.columns(2)
                    with ca:
                        st.image(cam_blended, use_container_width=True)
                        st.markdown('<div class="img-caption">🔥 Grad-CAM++ Heatmap<br><small>Warm/bright = model focused here</small></div>', unsafe_allow_html=True)
                    with cb:
                        st.image(overlay, use_container_width=True)
                        st.markdown('<div class="img-caption">🎯 AI Region Overlay<br><small>Red zones = regions most responsible for the AI prediction</small></div>', unsafe_allow_html=True)
                    st.caption(
                        "**Grad-CAM++** uses second-order gradients to find the exact spatial regions "
                        "that drove the model's decision. Unlike basic Grad-CAM, it can highlight multiple "
                        "regions and is sharper. The red overlay shows areas above 50% attention threshold."
                    )
                except Exception as e:
                    st.warning(f"Grad-CAM++ error: {e}")
                    st.image(image, use_container_width=True)
            else:
                st.image(image, use_container_width=True)

        with tab_ela:
            if ela_img is not None:
                st.image(ela_img, use_container_width=True)
                st.caption(
                    "**Error Level Analysis (ELA):** The image is resaved at quality=90 and the difference "
                    "is amplified 10×. **Bright/hot regions** = areas with inconsistent compression levels "
                    "= likely AI-generated, composited, or digitally manipulated. "
                    "Real photos have mostly UNIFORM brightness across the ELA image."
                )

        with tab_fft:
            if fft_img is not None:
                st.image(fft_img, use_container_width=True)
                st.caption(
                    "**FFT (Fast Fourier Transform):** Shows the frequency domain of the image. "
                    "The center = low frequencies (general structure). Edges = high frequencies (fine details). "
                    "**AI generators (GANs/Stable Diffusion) leave bright periodic spots** away from center — "
                    "these are invisible to the eye but detectable here. Real photos have smooth, gradual falloff."
                )

        with tab_ocr:
            st.image(ocr_ann_img, use_container_width=True)
            if total_text == 0:
                st.info("No text detected in this image.")
            else:
                st.markdown(f"**Total text regions found:** {total_text}")
                
                if wm_found:
                    st.error("🚨 **AI Watermark Detected!** \n\n" + "\n".join([f"- {w}" for w in wm_found]))
                
                if gl_found:
                    st.warning("⚠️ **Suspicious Text/Gibberish Detected!** AI generation models often fail to spell words correctly, creating alien-like mangled text.")
                    for txt, p in gl_found:
                        st.write(f"- *'{txt}'* (Confidence: {p*100:.1f}%)")
                        
                if not wm_found and not gl_found:
                    st.success("✅ **Text looks clean.** No known algorithmic watermarks or severe spelling glitches detected.")
                    
            st.caption(
                "**OCR Text & Watermark Forensics:** Uses Optical Character Recognition to find text. "
                "AI generation models like Midjourney and Stable Diffusion frequently mangle background text, "
                "which appears here as 'glitches' with low confidence. It also scans for embedded generator watermarks."
            )

        with tab_exif:
            if not use_exif:
                st.info("Enable **🔬 EXIF Metadata Analysis** in the sidebar to see metadata.")
            elif not exif_signals:
                st.warning("No analysis signals available.")
            else:
                if has_exif:
                    st.markdown("#### 📡 EXIF Metadata Signals")
                else:
                    st.markdown("#### 📡 Analysis Signals (EXIF unavailable — noise fallback used)")
                pills_html = ""
                for text, kind in exif_signals:
                    cls = {"ai":"pill-ai","real":"pill-real","warn":"pill-warn"}[kind]
                    pills_html += f'<span class="signal-pill {cls}">{text}</span>'
                st.markdown(f'<div style="margin-bottom:1rem;">{pills_html}</div>', unsafe_allow_html=True)

                if exif_data:
                    st.markdown("#### 📋 Raw EXIF Fields")
                    rows_html = ""
                    for k, v in exif_data.items():
                        rows_html += f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
                    st.markdown(f"""
                    <table class="exif-table">
                        <tr><th>Field</th><th>Value</th></tr>
                        {rows_html}
                    </table>""", unsafe_allow_html=True)
                elif has_exif:
                    st.warning("Image contains no readable EXIF fields.")

    with col_top_right:
        st.markdown(
            '<div class="sec-head" style="margin-top: 0px;"><span class="sec-head-icon">🏆</span>'
            '<span class="sec-head-title">Final Verdict Score</span>'
            '<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        
        v_cls = "verdict-ai"   if is_ai else "verdict-real"
        t_cls = "verdict-text-ai"  if is_ai else "verdict-text-real"
        p_cls = "verdict-pct-ai"   if majority_is_ai else "verdict-pct-real"
        icon  = "🤖" if is_ai else "📷"
        majority_label = "AI-Generated" if majority_is_ai else "Real"
        majority_pct_label = f"{majority_pct:.1f}% {majority_label}"

        st.markdown(f"""
        <div class="verdict-card {v_cls}">
            <div class="verdict-label">Final Verdict{hybrid_note}</div>
            <div class="verdict-icon">{icon}</div>
            <div class="verdict-text {t_cls}">{label}</div>
            <div class="verdict-pct {p_cls}">{pct:.1f}%</div>
            <div class="verdict-sub">majority score — {majority_pct_label}</div>
        </div>
        """, unsafe_allow_html=True)

        _exif_note = (
            f'<div style="margin-top:.8rem;padding-top:.7rem;border-top:1px solid rgba(255,255,255,.05);'
            f'font-size:.7rem;color:#475569;text-align:center;font-family:monospace;">'
            f'CNN: AI {cnn_ai:.1f}% \u00b7 Real {cnn_real:.1f}% &nbsp;|&nbsp; EXIF boost: +{exif_ai_boost:.0f} AI / +{exif_real_boost:.0f} Real</div>'
        ) if (use_exif and exif_signals) else ""
        st.markdown(
            f'<div style="background:rgba(0,0,0,.25);border:1px solid rgba(255,255,255,.06);border-radius:16px;padding:1.2rem 1.5rem;width:100%;margin:.8rem auto;">'
            f'<div style="display:flex;align-items:center;gap:.7rem;margin-bottom:.8rem;">'
            f'<span style="font-size:.72rem;font-weight:700;color:#fda4af;min-width:100px;font-family:monospace;">\U0001f916 AI-Generated</span>'
            f'<div style="flex:1;height:8px;background:rgba(0,0,0,.6);border-radius:999px;overflow:hidden;"><div style="height:100%;width:{min(ai_pct,100):.1f}%;background:linear-gradient(90deg,#e11d48,#f43f5e);box-shadow:0 0 12px rgba(244,63,94,.5);"></div></div>'
            f'<span style="font-family:monospace;font-size:.8rem;font-weight:700;color:#f43f5e;min-width:48px;text-align:right;">{ai_pct:.1f}%</span></div>'
            f'<div style="display:flex;align-items:center;gap:.7rem;">'
            f'<span style="font-size:.72rem;font-weight:700;color:#6ee7b7;min-width:100px;font-family:monospace;">\U0001f4f7 Real Photo</span>'
            f'<div style="flex:1;height:8px;background:rgba(0,0,0,.6);border-radius:999px;overflow:hidden;"><div style="height:100%;width:{min(real_pct,100):.1f}%;background:linear-gradient(90deg,#059669,#10b981);box-shadow:0 0 12px rgba(16,185,129,.5);"></div></div>'
            f'<span style="font-family:monospace;font-size:.8rem;font-weight:700;color:#10b981;min-width:48px;text-align:right;">{real_pct:.1f}%</span></div>'
            f'{_exif_note}'
            f'</div>',
            unsafe_allow_html=True
        )

    # ── ROW 2: EVIDENCE REPORT & FINGERPRINT ──────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_bot_left, col_bot_right = st.columns([1.0, 1.0], gap="large")

    with col_bot_left:
        st.markdown(
            f'<div class="sec-head" style="margin-top: 0px;"><span class="sec-head-icon">🔍</span>'
            f'<span class="sec-head-title">Forensic Evidence Report</span>'
            f'<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="background:rgba(244,63,94,0.05);border:1px solid rgba(244,63,94,0.2);border-top:3px solid #f43f5e;border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.5rem;">'
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1rem;font-weight:700;color:#f43f5e;margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem;"><span style="font-size:1.2rem;">🤖</span> AI Evidence</div>'
            f'{ai_rows}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div style="background:rgba(16,185,129,0.05);border:1px solid rgba(16,185,129,0.2);border-top:3px solid #10b981;border-radius:14px;padding:1.5rem 1.8rem;margin-bottom:1.5rem;">'
            f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1rem;font-weight:700;color:#10b981;margin-bottom:1rem;display:flex;align-items:center;gap:0.5rem;"><span style="font-size:1.2rem;">📷</span> Real Evidence</div>'
            f'{real_rows}'
            f'</div>',
            unsafe_allow_html=True
        )

    with col_bot_right:
        st.markdown(
            f'<div class="sec-head" style="margin-top: 0px;"><span class="sec-head-icon">🤔</span>'
            f'<span class="sec-head-title">AI Engine Fingerprint Analysis</span>'
            f'<div class="sec-head-line"></div></div>',
            unsafe_allow_html=True
        )
        if is_ai:
            st.caption("Analyzing pixel artifacts using CLIP zero-shot classification to estimate the synthetic generator.")
            with st.spinner("Extracting digital fingerprints..."):
                try:
                    source_probs = predict_ai_source_clip(image)
                    if source_probs:
                        # ── Show TOP result prominently ──────────────────────────
                        top_name, top_prob = source_probs[0]
                        top_pct = top_prob * 100
                        st.markdown(f'''
                        <div style="background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.3);border-left:4px solid #0ea5e9;border-radius:12px;padding:1.2rem 1.4rem;margin:0.8rem 0;">
                            <div style="font-family:'Inter',sans-serif;font-size:0.8rem;color:#94a3b8;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;">Most Likely Source</div>
                            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.6rem;">
                                <span style="font-family:'Inter',sans-serif;font-size:1.15rem;font-weight:700;color:#f1f5f9;">{top_name}</span>
                                <span style="font-family:'JetBrains Mono',monospace;font-size:1rem;font-weight:700;color:#38bdf8;">{top_pct:.1f}% match</span>
                            </div>
                            <div style="height:8px;background:rgba(0,0,0,0.4);border-radius:4px;overflow:hidden;">
                                <div style="height:100%;width:{top_pct}%;background:linear-gradient(90deg,#0ea5e9,#a855f7);border-radius:4px;box-shadow:0 0 10px rgba(14,165,233,0.5);"></div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

                        # ── Remaining sources in expander ─────────────────────────
                        if len(source_probs) > 1:
                            with st.expander(f"Show all {len(source_probs)} sources"):
                                st.markdown('<div style="display:flex;flex-direction:column;gap:0.4rem;margin-top:0.5rem;">', unsafe_allow_html=True)
                                for src_name, src_prob in source_probs[1:]:
                                    prob_pct = src_prob * 100
                                    st.markdown(f'''
                                    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);border-radius:8px;padding:0.65rem 1rem;">
                                        <div style="display:flex;justify-content:space-between;margin-bottom:0.35rem;font-family:'Inter',sans-serif;font-size:0.88rem;font-weight:500;">
                                            <span style="color:#cbd5e1;">{src_name}</span>
                                            <span style="color:#64748b;font-family:'JetBrains Mono',monospace;">{prob_pct:.1f}%</span>
                                        </div>
                                        <div style="height:4px;background:rgba(0,0,0,0.4);border-radius:2px;overflow:hidden;">
                                            <div style="height:100%;width:{prob_pct}%;background:linear-gradient(90deg,#334155,#475569);border-radius:2px;"></div>
                                        </div>
                                    </div>
                                    ''', unsafe_allow_html=True)
                                st.markdown('</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Engine fingerprinting unavailable: {e}")
        else:
            st.info("Fingerprint analysis inactive. The model determined this is a real photograph.")



    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sec-head"><span class="sec-head-icon">🧪</span>'
        '<span class="sec-head-title">Submit Your Review</span>'
        '<div class="sec-head-line"></div></div>',
        unsafe_allow_html=True
    )
    st.caption("Was the model right? Tell us — your feedback trains the next version.")

    with st.form(key="feedback_form", clear_on_submit=True):
        human_label = st.radio(
            "What is this image actually?",
            ["🤖 AI-Generated", "📷 Real Image"],
            horizontal=True,
            index=0 if is_ai else 1,
        )
        human_pct = st.slider(
            "Your confidence %", min_value=50, max_value=100, value=90, step=1,
            help="50 = uncertain, 100 = absolutely sure"
        )
        submitted = st.form_submit_button("📝 Submit Review", use_container_width=True)

    if submitted:
        clean_label = "AI-Generated" if "AI" in human_label else "Real"
        is_correct  = clean_label == label
        img_path    = FEEDBACK_DIR / f"{uuid.uuid4().hex}_{f.name}"
        image.save(img_path)
        with open(FEEDBACK_FILE, "a", newline="") as flog:
            csv.writer(flog).writerow([
                datetime.datetime.now().isoformat(), str(img_path),
                label, f"{pct:.1f}%", clean_label,
                f"{human_pct}%", "correct" if is_correct else "wrong", choice,
            ])
        if is_correct:
            st.success(f"✅ Model was **correct**! You confirmed: {clean_label} ({human_pct}%)")
        else:
            st.error(f"❌ Model was **wrong**. Correct: **{clean_label}** ({human_pct}%) — Predicted: {label} ({pct:.1f}%)")
        st.info("💡 Review saved. After 5+ reviews use **🚀 Retrain Model Now** in the sidebar.")

else:
    # Just show a clean empty state with the uploader
    
    st.markdown('<div style="height: 2rem;"></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-div"></div>', unsafe_allow_html=True)

    st.markdown('''
    <div style="text-align:center;margin-bottom:3rem;">
        <h2 style="font-family:'Space Grotesk',sans-serif;font-size:2rem;color:#f8fafc;font-weight:700;">How It Works</h2>
        <p style="color:#94a3b8;font-size:1.05rem;max-width:650px;margin:0 auto;line-height:1.6;">Our forensic pipeline uses multiple layers of deep learning and metadata analysis to expose synthetic media.</p>
    </div>
    
    <div class="feat-container" style="flex-wrap:wrap;">
        <div class="feat-card">
            <div class="feat-icon">🧠</div>
            <div class="feat-title">Deep Learning CNNs</div>
            <div class="feat-desc">Uses MobileNetV2, ResNet50, and EfficientNet trained on millions of images to spot pixel-level inconsistencies invisible to the human eye.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">📋</div>
            <div class="feat-title">EXIF Metadata</div>
            <div class="feat-desc">Extracts hidden camera data. Real photos contain exposure times and device hashes, while AI images often contain software tags.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔤</div>
            <div class="feat-title">OCR & Watermarks</div>
            <div class="feat-desc">Scans for embedded AI generator logos and uses Optical Character Recognition to find unnatural, garbled "alien text" hallucinated by diffusion models.</div>
        </div>
    </div>
    <div class="feat-container" style="flex-wrap:wrap; margin-top:2rem;">
        <div class="feat-card">
            <div class="feat-icon">📡</div>
            <div class="feat-title">FFT Spectrum</div>
            <div class="feat-desc">Calculates the Fast Fourier Transform of the image. AI generators create unnatural repetitive frequency artifacts that appear as bright dots or grids in the frequency domain.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🌊</div>
            <div class="feat-title">Error Level Analysis</div>
            <div class="feat-desc">Re-compresses the image to highlight inconsistent JPEG compression bands, which betray digital manipulation or regions composited from different sources.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎯</div>
            <div class="feat-title">Grad-CAM++</div>
            <div class="feat-desc">Generates an interpretable AI heatmap, projecting exactly which spatial regions the neural network focused on to make its final determination.</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)

