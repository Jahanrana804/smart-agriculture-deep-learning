"""
Task D — Ensemble weight optimisation on held-out validation set.
Finds the best CNN/ViT weights for the weighted ensemble.
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model as get_cnn_model
from task_c.vit_timm import get_vit_model


@torch.no_grad()
def collect_probs(model, dataloader, device, num_classes):
    """Run model over full dataloader, return (N, num_classes) probs + labels"""
    model.eval()
    all_probs  = []
    all_labels = []
    for imgs, labels in dataloader:
        imgs  = imgs.to(device)
        probs = torch.softmax(model(imgs), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def weighted_accuracy(weights, probs_list, labels):
    """Compute accuracy of weighted ensemble given weights list"""
    weights  = np.array(weights)
    weights  = weights / weights.sum()
    ensemble = sum(w * p for w, p in zip(weights, probs_list))
    preds    = ensemble.argmax(axis=1)
    return -np.mean(preds == labels)   # negative because we minimise


def optimise_ensemble_weights(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    save_dir    = cfg["paths"]["save_dir"]

    _, val_dl, _ = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        image_size  = cfg["data"]["image_size"],
    )

    models_to_try = [
        ("resnet50",  lambda: get_cnn_model("resnet50", num_classes)),
        ("custom",    lambda: CustomCNN(num_classes)),
        ("vit_b16",   lambda: get_vit_model("vit_b16",   num_classes, pretrained=False)),
        ("deit_small",lambda: get_vit_model("deit_small",num_classes, pretrained=False)),
        ("swin_tiny", lambda: get_vit_model("swin_tiny", num_classes, pretrained=False)),
    ]

    loaded_probs = []
    loaded_names = []

    for mname, loader_fn in models_to_try:
        ckpt = os.path.join(save_dir, f"{mname}_best.pt")
        if not os.path.exists(ckpt):
            continue
        print(f"  Loading {mname}...")
        try:
            m = loader_fn()
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m = m.to(device)
            probs, labels = collect_probs(m, val_dl, device, num_classes)
            loaded_probs.append(probs)
            loaded_names.append(mname)
            print(f"    Individual acc: {(probs.argmax(1) == labels).mean():.4f}")
        except Exception as e:
            print(f"    Failed: {e}")

    if len(loaded_probs) < 2:
        print("Need at least 2 models for ensemble optimisation.")
        return {"cnn": 0.5, "vit": 0.5}

    # optimise weights
    n = len(loaded_probs)
    x0     = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    result = minimize(
        weighted_accuracy,
        x0,
        args=(loaded_probs, labels),
        method="SLSQP",
        bounds=bounds,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
        options={"maxiter": 200, "ftol": 1e-9},
    )

    optimal_weights = result.x / result.x.sum()
    best_acc        = -result.fun

    print("\nOptimal ensemble weights:")
    weight_dict = {}
    for name, w in zip(loaded_names, optimal_weights):
        print(f"  {name}: {w:.4f}")
        weight_dict[name] = round(float(w), 4)

    print(f"Ensemble val accuracy: {best_acc:.4f}")

    # also compute simple equal-weight baseline
    equal_ensemble = sum(p for p in loaded_probs) / len(loaded_probs)
    equal_acc      = (equal_ensemble.argmax(1) == labels).mean()
    print(f"Equal-weight baseline: {equal_acc:.4f}")

    # save
    out_path = os.path.join(save_dir, "ensemble_weights.json")
    os.makedirs(save_dir, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(weight_dict, f, indent=2)
    print(f"\nWeights saved → {out_path}")

    return weight_dict


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    optimise_ensemble_weights(args.config)
