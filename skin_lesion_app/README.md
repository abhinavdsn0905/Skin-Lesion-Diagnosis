# Skin Lesion Classifier — Web App

A **Flask + HTML/CSS/JS** frontend for the HAM10000 Skin Lesion Classification project.

## Folder structure

```
skin_lesion_app/
├── app.py                  # Flask backend + inference logic
├── requirements.txt
├── models/                 # ← Place your .pth checkpoints here
│   ├── resnet50_best.pth
│   ├── efficientnet_b0_best.pth
│   └── dinov2_finetuned_best.pth
├── templates/
│   └── index.html          # Jinja2 HTML page
├── static/
│   ├── css/style.css       # All styles
│   └── js/app.js           # Frontend logic
└── uploads/                # Temp upload dir (auto-created)
```

## Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place model checkpoints in the models/ folder
#    (copy from your Google Drive: skin_lesion_project/models/)

# 4. Run the app
python app.py
```

Then open **http://localhost:5000** in your browser.

## Model checkpoints

| File | Model |
|---|---|
| `resnet50_best.pth` | ResNet-50 baseline |
| `efficientnet_b0_best.pth` | EfficientNet-B0 |
| `dinov2_finetuned_best.pth` | DINOv2 (partially fine-tuned) |

The app works **without checkpoints** — it shows random demo predictions so you can verify the UI before models are ready.

## Changing the checkpoint directory

Set the `MODEL_DIR` environment variable:

```bash
set MODEL_DIR=C:\path\to\your\models    # Windows CMD
$env:MODEL_DIR = "C:\path\to\models"   # PowerShell
python app.py
```

## Medical disclaimer

This application is for **educational research only** and has not been validated for clinical use.  
Do not use it to diagnose or rule out skin cancer — always consult a qualified dermatologist.
