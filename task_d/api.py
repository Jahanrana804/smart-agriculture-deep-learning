"""
Task D — FastAPI REST API
Accepts an uploaded image, returns top-3 predictions from:
  - Best CNN model
  - Best ViT model
  - Ensemble (majority vote + softmax average + weighted)
"""
import os
import sys
import io
import json
import yaml
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model as get_cnn_model
from task_c.vit_timm import get_vit_model

# ── Configuration ─────────────────────────────────────────────────────────────
CONFIG_PATH  = os.environ.get("CONFIG_PATH", "configs/config.yaml")
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

NUM_CLASSES = CFG["data"]["num_classes"]
IMAGE_SIZE  = CFG["data"]["image_size"]
SAVE_DIR    = CFG["paths"]["save_dir"]

# load class names from file if available, else use indices
CLASS_NAMES_FILE = "data/class_names.json"
if os.path.exists(CLASS_NAMES_FILE):
    with open(CLASS_NAMES_FILE) as f:
        CLASS_NAMES = json.load(f)
else:
    CLASS_NAMES = [f"class_{i}" for i in range(NUM_CLASSES)]

# ── Image preprocessing ───────────────────────────────────────────────────────
PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std= [0.229, 0.224, 0.225]),
])

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_safe(loader_fn, ckpt_path, label):
    """Load a model from checkpoint, return None if not found"""
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path} — {label} will be skipped")
        return None
    try:
        model = loader_fn()
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval().to(DEVICE)
        print(f"[OK] Loaded {label}")
        return model
    except Exception as e:
        print(f"[ERR] Failed to load {label}: {e}")
        return None

# best CNN — try resnet50 first, fallback to custom
cnn_ckpt   = os.path.join(SAVE_DIR, "resnet50_best.pt")
cnn_model  = load_model_safe(
    lambda: get_cnn_model("resnet50", NUM_CLASSES),
    cnn_ckpt, "ResNet-50"
)
if cnn_model is None:
    cnn_ckpt  = os.path.join(SAVE_DIR, "custom_best.pt")
    cnn_model = load_model_safe(
        lambda: CustomCNN(NUM_CLASSES),
        cnn_ckpt, "Custom CNN"
    )

# best ViT — try vit_b16 first
vit_ckpt  = os.path.join(SAVE_DIR, "vit_b16_best.pt")
vit_model = load_model_safe(
    lambda: get_vit_model("vit_b16", NUM_CLASSES, pretrained=False),
    vit_ckpt, "ViT-B/16"
)
if vit_model is None:
    vit_ckpt  = os.path.join(SAVE_DIR, "deit_small_best.pt")
    vit_model = load_model_safe(
        lambda: get_vit_model("deit_small", NUM_CLASSES, pretrained=False),
        vit_ckpt, "DeiT-Small"
    )

# ensemble weights (optimised on val set, or equal if not available)
ENSEMBLE_WEIGHTS_FILE = os.path.join(SAVE_DIR, "ensemble_weights.json")
if os.path.exists(ENSEMBLE_WEIGHTS_FILE):
    with open(ENSEMBLE_WEIGHTS_FILE) as f:
        ensemble_weights = json.load(f)
    W_CNN = ensemble_weights.get("cnn", 0.5)
    W_VIT = ensemble_weights.get("vit", 0.5)
else:
    W_CNN, W_VIT = 0.5, 0.5

# ── Inference helpers ─────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return PREPROCESS(img).unsqueeze(0).to(DEVICE)


@torch.no_grad()
def get_probs(model, inp: torch.Tensor) -> torch.Tensor:
    """Returns softmax probabilities (1, num_classes)"""
    return F.softmax(model(inp), dim=1)


def top3(probs: torch.Tensor):
    """Returns list of {class, confidence} for top 3"""
    top_probs, top_idx = probs[0].topk(3)
    return [
        {
            "class":      CLASS_NAMES[i.item()],
            "confidence": round(p.item(), 4),
        }
        for p, i in zip(top_probs, top_idx)
    ]


def majority_vote(cnn_probs, vit_probs):
    """Majority vote ensemble — pick class with most votes"""
    cnn_pred = cnn_probs.argmax(1)
    vit_pred = vit_probs.argmax(1)
    if cnn_pred == vit_pred:
        return cnn_pred.item()
    # tie: pick the one with higher max probability
    return (cnn_probs.max() > vit_probs.max()).item() * cnn_pred.item() + \
           (cnn_probs.max() <= vit_probs.max()).item() * vit_pred.item()


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "AgriDL Inference API",
    description = "Plant disease detection — CNN + ViT + Ensemble",
    version     = "1.0",
)


@app.get("/health")
def health_check():
    return {
        "status":    "ok",
        "cnn_ready": cnn_model is not None,
        "vit_ready": vit_model is not None,
        "device":    DEVICE,
    }


@app.post("/predict/cnn")
async def predict_cnn(file: UploadFile = File(...)):
    if cnn_model is None:
        raise HTTPException(status_code=503, detail="CNN model not loaded")
    img_bytes = await file.read()
    inp   = preprocess_image(img_bytes)
    probs = get_probs(cnn_model, inp)
    return JSONResponse({"model": "cnn", "top3": top3(probs)})


@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    if vit_model is None:
        raise HTTPException(status_code=503, detail="ViT model not loaded")
    img_bytes = await file.read()
    inp   = preprocess_image(img_bytes)
    probs = get_probs(vit_model, inp)
    return JSONResponse({"model": "vit", "top3": top3(probs)})


@app.post("/predict/ensemble")
async def predict_ensemble(file: UploadFile = File(...)):
    img_bytes = await file.read()
    inp       = preprocess_image(img_bytes)
    results   = {}

    # collect available model probs
    available_probs = []
    if cnn_model is not None:
        cnn_probs = get_probs(cnn_model, inp)
        results["cnn_top3"] = top3(cnn_probs)
        available_probs.append(("cnn", cnn_probs, W_CNN))

    if vit_model is not None:
        vit_probs = get_probs(vit_model, inp)
        results["vit_top3"] = top3(vit_probs)
        available_probs.append(("vit", vit_probs, W_VIT))

    if not available_probs:
        raise HTTPException(status_code=503, detail="No models loaded")

    # (i) Majority vote
    if len(available_probs) >= 2:
        vote_class = majority_vote(cnn_probs, vit_probs)
        results["majority_vote"] = CLASS_NAMES[vote_class]

    # (ii) Average softmax
    avg_probs = sum(p for _, p, _ in available_probs) / len(available_probs)
    results["softmax_avg_top3"] = top3(avg_probs)

    # (iii) Weighted ensemble
    total_w    = sum(w for _, _, w in available_probs)
    weighted   = sum(p * (w / total_w) for _, p, w in available_probs)
    results["weighted_ensemble_top3"] = top3(weighted)

    return JSONResponse(results)


@app.post("/predict/all")
async def predict_all(file: UploadFile = File(...)):
    """Single endpoint returning predictions from all models"""
    img_bytes = await file.read()
    inp       = preprocess_image(img_bytes)
    out       = {}

    if cnn_model is not None:
        out["cnn"] = top3(get_probs(cnn_model, inp))
    if vit_model is not None:
        out["vit"] = top3(get_probs(vit_model, inp))

    # weighted ensemble
    if cnn_model and vit_model:
        cp = get_probs(cnn_model, inp)
        vp = get_probs(vit_model, inp)
        ensemble = W_CNN * cp + W_VIT * vp
        out["ensemble"] = top3(ensemble)

    return JSONResponse(out)


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("task_d.api:app", host="0.0.0.0", port=8000, reload=False)
