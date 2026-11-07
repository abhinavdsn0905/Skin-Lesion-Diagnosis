"""
Skin Lesion Classification – Flask Backend
==========================================
Loads the three trained model checkpoints (ResNet-50, EfficientNet-B0,
DINOv2 fine-tuned) and serves predictions through a REST endpoint.

Endpoints
---------
GET  /            → main UI
POST /predict     → JSON { image, models[] } → JSON { results[] }
GET  /health      → { status: "ok", loaded_models: [...] }
"""

import os
import io
import base64
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template, request
from PIL import Image
from torchvision import models, transforms

# ---------------------------------------------------------------------------
# Optional DINOv2 (transformers library)
# ---------------------------------------------------------------------------
try:
    from transformers import AutoModel
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit
UPLOAD_FOLDER = Path(__file__).parent / "uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Inference device: %s", DEVICE)

# ---------------------------------------------------------------------------
# Class metadata
# ---------------------------------------------------------------------------
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / Intraepithelial carcinoma",
    "bcc":   "Basal cell carcinoma",
    "bkl":   "Benign keratosis-like lesions",
    "df":    "Dermatofibroma",
    "mel":   "Melanoma",
    "nv":    "Melanocytic nevi",
    "vasc":  "Vascular lesions",
}
CLASS_RISK = {
    "akiec": "high",
    "bcc":   "high",
    "bkl":   "low",
    "df":    "low",
    "mel":   "high",
    "nv":    "low",
    "vasc":  "medium",
}

# Plain-English patient/user reviews for each category
CLASS_REVIEWS = {
    "mel": {
        "title": "Suspected Melanoma (Malignant)",
        "summary": "The model detects visual patterns characteristic of Melanoma, an aggressive form of skin cancer arising from melanocytes.",
        "clinical_advice": "Urgent dermatologist review recommended. An in-person dermoscopic exam and potential excisional biopsy are standard protocol.",
        "watch_signs": "Check the ABCDE criteria: Asymmetry, Border irregularity, Color variegation, Diameter > 6mm, and Evolution over time."
    },
    "bcc": {
        "title": "Suspected Basal Cell Carcinoma (Malignant)",
        "summary": "Visual features align with Basal Cell Carcinoma, the most frequent skin malignancy, typically presenting with pearly translucency or arborizing vessels.",
        "clinical_advice": "Prompt dermatological assessment advised. While BCC rarely metastasizes, early treatment avoids deeper local tissue invasion.",
        "watch_signs": "Look out for persistent sores that bleed, crust over, or fail to heal over several weeks."
    },
    "akiec": {
        "title": "Suspected Actinic Keratosis / Bowen's Disease (Pre-cancerous)",
        "summary": "Features indicate Actinic Keratosis or Intraepithelial Carcinoma, common sun-induced rough patches that carry a potential to progress to squamous cell carcinoma.",
        "clinical_advice": "Consult a dermatologist for routine evaluation. These are commonly treated quickly with cryotherapy (freezing) or topical creams.",
        "watch_signs": "Feel for rough, scaly, sandpapery texture on frequently sun-exposed areas."
    },
    "nv": {
        "title": "Benign Melanocytic Nevus (Common Mole)",
        "summary": "The lesion displays consistent, regular pigment patterns typical of a harmless common mole or beauty spot composed of benign melanocytes.",
        "clinical_advice": "Generally no immediate intervention is required. Continue routine periodic self-monitoring at home.",
        "watch_signs": "Consult a physician if you observe sudden changes in size, shape, symmetry, dark colors, or bleeding."
    },
    "bkl": {
        "title": "Benign Keratosis-like Lesion (Harmless)",
        "summary": "Appearance resembles benign seborrheic keratosis, solar lentigo, or lichenoid keratosis. These are non-cancerous skin occurrences common with age and sun exposure.",
        "clinical_advice": "No urgent clinical concern. A routine skin check can confirm this diagnosis if the lesion causes itching or irritation.",
        "watch_signs": "Watch for sudden irregular pigmentation, ulceration, or rapid growth."
    },
    "df": {
        "title": "Dermatofibroma (Benign Fibrous Nodule)",
        "summary": "Features match a dermatofibroma, a non-cancerous firm papule or nodule often triggered by a minor injury, scratch, or bug bite.",
        "clinical_advice": "Harmless skin lesion. Medical removal is usually purely elective or for cosmetic reasons.",
        "watch_signs": "Typically presents the classic 'dimple sign' when pinched lightly from the sides."
    },
    "vasc": {
        "title": "Vascular Lesion (Blood Vessel Proliferation)",
        "summary": "Presents dermoscopic vascular structures such as red/purple lacunae, characteristic of cherry angiomas, hemangiomas, or pyogenic granulomas.",
        "clinical_advice": "Generally benign. Evaluation is suggested if the spot bleeds recurrently, changes rapidly, or is subjected to frequent friction.",
        "watch_signs": "Monitor for unexpected spontaneous bleeding or rapid size increase."
    }
}

# ---------------------------------------------------------------------------
# Model paths – update these if your checkpoints live elsewhere
# ---------------------------------------------------------------------------
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).parent / "models"))

CHECKPOINT_PATHS = {
    "ResNet-50":        MODEL_DIR / "resnet50_best.pth",
    "EfficientNet-B0":  MODEL_DIR / "efficientnet_b0_best.pth",
    "DINOv2":           MODEL_DIR / "dinov2_finetuned_best.pth",
}

# ---------------------------------------------------------------------------
# DINOv2 wrapper (must match training architecture)
# ---------------------------------------------------------------------------
class DINOv2Classifier(nn.Module):
    def __init__(self, backbone, num_classes: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.config.hidden_size, num_classes)

    def forward(self, images):
        outputs = self.backbone(pixel_values=images)
        return self.head(outputs.last_hidden_state[:, 0, :])


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------
LOADED_MODELS: dict[str, nn.Module] = {}


def _load_checkpoint(path: Path):
    """Load a .pth checkpoint safely."""
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=DEVICE, weights_only=False)


def load_all_models():
    """Try to load each checkpoint; skip gracefully if missing."""
    # -- ResNet-50 --
    try:
        ckpt = _load_checkpoint(CHECKPOINT_PATHS["ResNet-50"])
        num_classes = len(ckpt.get("class_names", CLASS_NAMES))
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        m.load_state_dict(ckpt["model_state_dict"])
        LOADED_MODELS["ResNet-50"] = m.to(DEVICE).eval()
        log.info("ResNet-50 loaded ✓")
    except Exception as exc:
        log.warning("ResNet-50 not loaded: %s", exc)

    # -- EfficientNet-B0 --
    try:
        ckpt = _load_checkpoint(CHECKPOINT_PATHS["EfficientNet-B0"])
        num_classes = len(ckpt.get("class_names", CLASS_NAMES))
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
        m.load_state_dict(ckpt["model_state_dict"])
        LOADED_MODELS["EfficientNet-B0"] = m.to(DEVICE).eval()
        log.info("EfficientNet-B0 loaded ✓")
    except Exception as exc:
        log.warning("EfficientNet-B0 not loaded: %s", exc)

    # -- DINOv2 fine-tuned --
    if TRANSFORMERS_AVAILABLE:
        try:
            ckpt = _load_checkpoint(CHECKPOINT_PATHS["DINOv2"])
            num_classes = len(ckpt.get("class_names", CLASS_NAMES))
            backbone = AutoModel.from_pretrained("facebook/dinov2-base")
            m = DINOv2Classifier(backbone, num_classes)
            m.load_state_dict(ckpt["model_state_dict"])
            LOADED_MODELS["DINOv2"] = m.to(DEVICE).eval()
            log.info("DINOv2 fine-tuned loaded ✓")
        except Exception as exc:
            log.warning("DINOv2 not loaded: %s", exc)
    else:
        log.warning("transformers not installed – DINOv2 unavailable")

    if not LOADED_MODELS:
        log.warning(
            "No model checkpoints were found in %s. "
            "The app will return demo (random) predictions until real "
            "checkpoints are placed there.",
            MODEL_DIR,
        )


load_all_models()

# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------
INFERENCE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    img = pil_image.convert("RGB")
    return INFERENCE_TRANSFORM(img).unsqueeze(0).to(DEVICE)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(model: nn.Module, tensor: torch.Tensor) -> np.ndarray:
    logits = model(tensor)
    probs = torch.softmax(logits.float(), dim=1)[0].cpu().numpy()
    return probs


def demo_prediction() -> np.ndarray:
    """Return random probabilities when no real model is available."""
    raw = np.random.dirichlet(np.ones(len(CLASS_NAMES)))
    return raw


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Landing page: Project overview, clinical taxonomy, workflow, and model highlights."""
    return render_template("landing.html")


@app.route("/analyze")
def analyze():
    """Dedicated image upload and analysis workspace with plain-English report overview."""
    available = ["ResNet-50", "EfficientNet-B0", "DINOv2"]
    return render_template("analysis.html", available_models=available, loaded_models=list(LOADED_MODELS.keys()))


@app.route("/results")
def results():
    """Full research report: metrics, graphs, per-class F1, and training curves."""
    return render_template("results.html")


@app.route("/about")
def about():
    """Redirect to the comprehensive landing page."""
    return render_template("landing.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "loaded_models": list(LOADED_MODELS.keys()),
                    "device": str(DEVICE)})


@app.route("/predict", methods=["POST"])
def predict():
    # ---- parse request ----
    if request.is_json:
        data = request.get_json()
        image_b64 = data.get("image")          # base64-encoded data URL
        selected_models = data.get("models", list(LOADED_MODELS.keys()))
    else:
        image_b64 = request.form.get("image")
        selected_models = request.form.getlist("models")

    if not image_b64:
        return jsonify({"error": "No image provided."}), 400

    # ---- decode image ----
    try:
        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        image_bytes = base64.b64decode(image_b64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        return jsonify({"error": f"Invalid image: {exc}"}), 400

    # ---- run inference ----
    tensor = preprocess_image(pil_image)
    results = []

    for model_name in selected_models:
        if model_name not in LOADED_MODELS:
            # Fallback: demo random prediction
            probs = demo_prediction()
            note = "demo (checkpoint not loaded)"
        else:
            probs = run_inference(LOADED_MODELS[model_name], tensor)
            note = ""

        best_idx = int(np.argmax(probs))
        predicted_code = CLASS_NAMES[best_idx]
        review_info = CLASS_REVIEWS.get(predicted_code, {
            "title": CLASS_DESCRIPTIONS[predicted_code],
            "summary": "Differential classification output for this category.",
            "clinical_advice": "Consult a healthcare specialist for examination.",
            "watch_signs": "Monitor for visual changes over time."
        })

        class_scores = [
            {
                "label": CLASS_NAMES[i],
                "description": CLASS_DESCRIPTIONS[CLASS_NAMES[i]],
                "risk": CLASS_RISK[CLASS_NAMES[i]],
                "score": float(round(probs[i], 6)),
                "percent": f"{probs[i]:.1%}",
            }
            for i in range(len(CLASS_NAMES))
        ]
        # Sort descending
        class_scores.sort(key=lambda x: x["score"], reverse=True)

        results.append({
            "model": model_name,
            "predicted_label": predicted_code,
            "predicted_description": CLASS_DESCRIPTIONS[predicted_code],
            "predicted_risk": CLASS_RISK[predicted_code],
            "confidence": f"{probs[best_idx]:.1%}",
            "scores": class_scores,
            "review": review_info,
            "note": note,
        })

    return jsonify({"results": results})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
