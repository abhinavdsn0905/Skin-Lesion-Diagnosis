# DermaAI — Skin Lesion Classification

> **CSE-471 Deep Learning Research Project**  
> Comparative study of CNN baselines (ResNet-50, EfficientNet-B0) vs. self-supervised Vision Transformers (DINOv2) on the HAM10000 dermoscopic image dataset.

## 🌐 Live App Pages

| Page | Description |
|------|-------------|
| `/` | Landing page — project overview, 7 diagnostic classes, methodology |
| `/analyze` | Interactive classifier — upload image → multi-model AI diagnosis |
| `/results` | Research results — charts, metrics table, heatmaps |
| `/about` | About the project |

## 📊 Benchmark Results

| Model | Test Acc | Balanced Acc | Macro-F1 | Macro-AUROC |
|-------|----------|-------------|----------|-------------|
| ResNet-50 | **82.42%** | 65.53% | **65.19%** | 95.58% |
| EfficientNet-B0 | 81.33% | **65.65%** | 64.31% | **95.70%** |
| DINOv2 Frozen | 73.93% | 62.46% | 58.77% | 91.81% |
| DINOv2 Fine-tuned | 75.82% | 63.42% | 59.20% | 93.74% |

## 🗂️ Project Structure

```
CSE-471/
├── Skin_Lesion_Classification_Organized.ipynb   ← Full training notebook
├── skin_lesion_app/
│   ├── app.py                   ← Flask backend
│   ├── requirements.txt         ← Python dependencies
│   ├── templates/
│   │   ├── landing.html         ← Home page
│   │   ├── analysis.html        ← Image analysis workspace
│   │   ├── results.html         ← Research results
│   │   └── about.html           ← About
│   ├── static/
│   │   ├── css/                 ← Stylesheets
│   │   ├── js/app.js            ← Frontend logic
│   │   └── img/                 ← Generated result charts
│   ├── models/                  ← Place .pth checkpoints here (see below)
│   └── uploads/                 ← Temp image storage (auto-cleared)
├── HAM10000_images_part_1/      ← Dataset images [NOT in git — too large]
├── HAM10000_images_part_2/      ← Dataset images [NOT in git — too large]
└── .gitignore
```

## 🚀 Setup & Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/CSE-471-Skin-Lesion.git
cd CSE-471-Skin-Lesion
```

### 2. Install dependencies
```bash
cd skin_lesion_app
pip install -r requirements.txt
```

### 3. Place model checkpoints
Download or train the following checkpoint files and place them in `skin_lesion_app/models/`:
- `resnet50_best.pth`
- `efficientnet_b0_best.pth`
- `dinov2_finetuned_best.pth`

> Without checkpoints the app still runs in **simulation mode** — all 3 models are selectable and return probabilistic outputs.

### 4. Run the Flask server
```bash
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

## 📁 Dataset

This project uses the **HAM10000 (Human Against Machine with 10,000 training images)** dataset.  
Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Place images in:
```
CSE-471/
├── HAM10000_images_part_1/
└── HAM10000_images_part_2/
```

## 🛠️ Tech Stack

- Python 3.10 · PyTorch 2.x · Torchvision
- Facebook DINOv2 (via HuggingFace Transformers)
- Flask · Jinja2 · HTML/CSS/JS
- Matplotlib · scikit-learn
- Google Colab (T4 GPU for training)

## ⚠️ Disclaimer

This is an **educational research prototype only**. Model predictions are softmax probability scores, not calibrated medical diagnoses. Always consult a qualified dermatologist for clinical evaluation.

---

*CSE-471 · Skin Lesion Classification Research · HAM10000 · Educational Use Only*

