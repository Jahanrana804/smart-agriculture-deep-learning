# CS4152 — Deep Learning Smart Agriculture Project

## Project Structure

```
agri_dl_project/
├── configs/config.yaml        ← all hyperparameters
├── utils/
│   ├── metrics.py             ← accuracy, F1, confusion matrix
│   └── plotting.py            ← all plot functions
├── task_a/
│   ├── dataset.py             ← data loading + augmentation
│   ├── custom_cnn.py          ← 5-block CNN from scratch
│   ├── transfer_learning.py   ← ResNet50, EfficientNet, VGG16, MobileNetV2, DenseNet121
│   ├── train.py               ← training loop
│   ├── evaluate.py            ← full eval + latency benchmark
│   ├── gradcam.py             ← Grad-CAM visualisation
│   ├── export_onnx.py         ← ONNX export + INT8 quantisation
│   └── ablation.py            ← ablation study runner
├── task_b/
│   ├── dcgan.py               ← DCGAN for minority class
│   ├── cgan.py                ← Conditional GAN
│   ├── ddpm.py                ← DDPM with U-Net backbone
│   ├── fid_score.py           ← FID + Inception Score
│   └── downstream.py          ← downstream impact evaluation
├── task_c/
│   ├── vit_timm.py            ← ViT-B/16, DeiT-Small, Swin-Tiny, Hybrid CNN-ViT
│   ├── attention_viz.py       ← attention rollout + entropy
│   └── benchmark.py           ← full benchmark table
├── task_d/
│   ├── api.py                 ← FastAPI REST server
│   ├── ensemble.py            ← ensemble weight optimisation
│   ├── calibration.py         ← ECE + reliability diagrams
│   └── deployment_analysis.py ← distribution shift + energy footprint
├── run_all.py                 ← master pipeline script
├── setup_data.py              ← one-time dataset setup
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 1. Environment Setup

```bash
conda create -n agri_dl python=3.11 -y
conda activate agri_dl

# PyTorch for RTX 5070 (CUDA 12.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# all other dependencies
pip install -r requirements.txt
```

---

## 2. Download Dataset

### Option A — Kaggle CLI (recommended)
```bash
pip install kaggle
# put kaggle.json in ~/.kaggle/ (from kaggle.com → Settings → API)
chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d vipoooool/new-plant-diseases-dataset \
    -p data/plantvillage --unzip
```

### Option B — Git clone (smaller version)
```bash
git clone https://github.com/spMohanty/PlantVillage-Dataset data/plantvillage_raw
```

### One-time setup (saves class names, updates config)
```bash
python setup_data.py --train_dir data/plantvillage/train
```

---

## 3. Run Individual Tasks

### Task A — CNN Baseline
```bash
# train single model
python -m task_a.train --model custom  --aug medium
python -m task_a.train --model resnet50 --aug heavy
python -m task_a.train --model efficientnet_b3 --aug medium
python -m task_a.train --model mobilenet_v2 --aug light

# evaluate + benchmark
python -m task_a.evaluate --model resnet50

# grad-cam
python -m task_a.gradcam --model resnet50

# edge deployment / ONNX export
python -m task_a.export_onnx --model resnet50

# ablation study (quick, 5 epochs)
python -m task_a.ablation --epochs 5
```

### Task B — GANs + Diffusion
```bash
# DCGAN on minority class (index 0)
python -m task_b.dcgan --target_class 0

# Conditional GAN
python -m task_b.cgan

# DDPM
python -m task_b.ddpm

# FID + IS scores (run after training)
python -m task_b.fid_score

# Downstream impact evaluation
python -m task_b.downstream
```

### Task C — Vision Transformers
```bash
# train with pretrained weights (recommended)
python -m task_c.vit_timm --model vit_b16
python -m task_c.vit_timm --model deit_small --mixup
python -m task_c.vit_timm --model swin_tiny
python -m task_c.vit_timm --model hybrid

# train WITHOUT pretrained (for comparison)
python -m task_c.vit_timm --model vit_b16 --no_pretrain

# attention rollout visualisation
python -m task_c.attention_viz --model vit_b16 --n 8

# full benchmark table (needs all models trained)
python -m task_c.benchmark
```

### Task D — Deployment
```bash
# optimise ensemble weights
python -m task_d.ensemble

# calibration analysis (ECE + reliability diagrams)
python -m task_d.calibration

# distribution shift + energy footprint
python -m task_d.deployment_analysis

# start API server
python -m task_d.api
# API is now live at http://localhost:8000
# Docs at http://localhost:8000/docs

# Docker
docker-compose up --build
```

---

## 4. Run Everything (full pipeline)

```bash
python run_all.py --task all
# or individual tasks:
python run_all.py --task a
python run_all.py --task b
python run_all.py --task c
python run_all.py --task d
```

---

## 5. API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Check if models are loaded |
| `/predict/cnn` | POST | CNN prediction (top 3) |
| `/predict/vit` | POST | ViT prediction (top 3) |
| `/predict/ensemble` | POST | Ensemble (majority + softmax avg + weighted) |
| `/predict/all` | POST | All models in one call |

```bash
# example call
curl -X POST http://localhost:8000/predict/all \
     -F "file=@my_leaf_image.jpg"
```

---

## 6. Outputs

| Location | Contents |
|---|---|
| `checkpoints/` | Best model weights (.pt) and ONNX exports |
| `logs/` | Training curves, confusion matrices, Grad-CAMs, attention maps, reliability diagrams, benchmark CSV |
| `samples/` | GAN/DDPM generated image grids |

---

## 7. Hardware

Tested on RTX 5070 (12GB VRAM, CUDA 12.8).
Expected training times:
- Task A custom CNN: ~20 min (25 epochs)
- Task A ResNet-50: ~15 min (25 epochs)
- Task C ViT-B/16: ~45 min (20 epochs)
- Task B DCGAN: ~30 min (100 epochs)
- Task B DDPM: ~2–3 hours (50 epochs)

---

## 8. LLM Tool Disclosure (Academic Integrity)

As required by the spec, this project used AI coding assistance (Claude) for:
- Boilerplate code structure and file organisation
- Standard PyTorch training loop templates
- FastAPI endpoint scaffolding

All architectural decisions, hyperparameter choices, experimental design,
analysis, and written report sections are the students' own work.

---

*CS4152 Deep Learning & Neural Networks — Spring 2026*
*Group: [Your names] | Submitted: Week 14*
