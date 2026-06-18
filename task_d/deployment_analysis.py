"""
Task D — Distribution shift simulation and energy footprint analysis.
Tests best model on out-of-distribution images (PlantDoc as OOD for PlantVillage).
Estimates kWh per 1000 inferences for GPU / CPU / quantised CPU.
"""
import os
import sys
import time
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model as get_cnn_model
from task_c.vit_timm import get_vit_model
from utils.metrics import evaluate_model


# ── Distribution Shift ────────────────────────────────────────────────────────
def apply_ood_transform(severity="medium"):
    """
    Simulate OOD conditions: different lighting, blur, noise.
    severity: light | medium | heavy
    """
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    if severity == "light":
        ood = [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    elif severity == "medium":
        ood = [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.5, 2.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:  # heavy
        ood = [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.8, contrast=0.8,
                                   saturation=0.8, hue=0.3),
            transforms.GaussianBlur(kernel_size=9, sigma=(1.0, 4.0)),
            transforms.RandomGrayscale(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
        ]

    return transforms.Compose(base), transforms.Compose(ood)


def run_distribution_shift(model, val_dir, device, num_classes,
                            log_dir="logs", model_name="model"):
    """
    Tests model under clean + 3 OOD severity levels.
    Reports accuracy degradation at each level.
    """
    results = {}
    severities = ["clean", "light", "medium", "heavy"]

    for sev in severities:
        if sev == "clean":
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            _, tf = apply_ood_transform(sev)

        ds = datasets.ImageFolder(val_dir, transform=tf)
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)

        metrics = evaluate_model(model, dl, device, num_classes)
        acc     = metrics["accuracy"]
        results[sev] = acc
        print(f"  [{sev:6s}] Accuracy: {acc:.4f}")

    # plot accuracy degradation
    accs   = [results[s] for s in severities]
    labels = ["Clean", "Light OOD", "Medium OOD", "Heavy OOD"]

    plt.figure(figsize=(7, 4))
    colors = ["steelblue", "orange", "tomato", "darkred"]
    bars   = plt.bar(labels, accs, color=colors, width=0.5, alpha=0.85)
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.title(f"Distribution Shift — {model_name}")
    plt.grid(True, axis="y", alpha=0.3)
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.01,
                 f"{acc:.3f}", ha="center", fontsize=9)
    plt.tight_layout()

    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{model_name}_ood_shift.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved OOD plot → {path}")

    # degradation summary
    clean_acc = results["clean"]
    print(f"\n  Accuracy degradation from clean baseline ({clean_acc:.4f}):")
    for sev in ["light", "medium", "heavy"]:
        drop = clean_acc - results[sev]
        print(f"    {sev:6s}: -{drop:.4f}  ({drop/clean_acc*100:.1f}% relative)")

    return results


# ── Energy Footprint ──────────────────────────────────────────────────────────
def measure_inference_time_ms(model, device, image_size=224, n=500):
    """Return mean inference time in ms for a single image"""
    model.eval().to(device)
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(20): _ = model(dummy)   # warmup
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): _ = model(dummy)
    return (time.perf_counter() - start) / n * 1000


def estimate_energy_kwh(ms_per_image, tdp_watts, n_inferences=1000):
    """
    Energy estimate: (time_hours) * TDP_watts / 1000 = kWh
    tdp_watts: typical TDP of the device
    """
    time_hours = (ms_per_image * n_inferences) / (1000 * 3600)
    return time_hours * tdp_watts


def run_energy_analysis(model, model_name, image_size=224, log_dir="logs"):
    """
    Estimates kWh per 1000 inferences for:
      - GPU (RTX 5070, ~150W TDP)
      - CPU (Intel Core i7, ~65W TDP)
      - Quantised CPU (~65W, faster inference)
    """
    print(f"\n  Energy analysis: {model_name}")

    # GPU
    gpu_tdp = 150   # RTX 5070 ~150W
    cpu_tdp = 65    # typical laptop/desktop CPU

    rows = []

    if torch.cuda.is_available():
        lat_gpu = measure_inference_time_ms(model, "cuda", image_size)
        kwh_gpu = estimate_energy_kwh(lat_gpu, gpu_tdp)
        rows.append(("GPU (RTX 5070)", lat_gpu, gpu_tdp, kwh_gpu))

    lat_cpu = measure_inference_time_ms(model.cpu(), "cpu", image_size)
    kwh_cpu = estimate_energy_kwh(lat_cpu, cpu_tdp)
    rows.append(("CPU (FP32)", lat_cpu, cpu_tdp, kwh_cpu))

    # INT8 quantised CPU
    try:
        q_model = torch.quantization.quantize_dynamic(
            model.cpu(),
            {torch.nn.Linear, torch.nn.Conv2d},
            dtype=torch.qint8
        )
        lat_q = measure_inference_time_ms(q_model, "cpu", image_size)
        kwh_q = estimate_energy_kwh(lat_q, cpu_tdp)
        rows.append(("CPU (INT8 quantised)", lat_q, cpu_tdp, kwh_q))
    except Exception:
        pass

    # print table
    print(f"\n  {'Device':<25} {'Lat (ms)':>10} {'TDP (W)':>8} {'kWh/1k':>10}")
    print("  " + "-"*57)
    for name, lat, tdp, kwh in rows:
        print(f"  {name:<25} {lat:>10.2f} {tdp:>8} {kwh:>10.6f}")
    print()

    # Green AI note
    best_device = min(rows, key=lambda r: r[3])
    print(f"  Most energy-efficient: {best_device[0]} "
          f"({best_device[3]:.6f} kWh per 1000 inferences)")
    print("  Per Green AI principles: prefer quantised CPU/edge deployment")
    print("  for low-volume inference; GPU only justified for batch processing.")

    os.makedirs(log_dir, exist_ok=True)
    # bar chart
    devices = [r[0] for r in rows]
    kwhs    = [r[3] for r in rows]
    plt.figure(figsize=(7, 4))
    plt.bar(devices, kwhs, color=["#2196F3", "#FF9800", "#4CAF50"][:len(rows)], alpha=0.85)
    plt.ylabel("kWh per 1000 inferences")
    plt.title(f"Energy Footprint — {model_name}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    path = os.path.join(log_dir, f"{model_name}_energy.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved energy chart → {path}")

    return rows


# ── Runner ────────────────────────────────────────────────────────────────────
def run_deployment_analysis(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    save_dir    = cfg["paths"]["save_dir"]
    log_dir     = cfg["paths"]["log_dir"]

    # pick best available model
    candidates = [
        ("resnet50",  lambda: get_cnn_model("resnet50", num_classes)),
        ("vit_b16",   lambda: get_vit_model("vit_b16", num_classes, pretrained=False)),
        ("deit_small",lambda: get_vit_model("deit_small", num_classes, pretrained=False)),
        ("custom",    lambda: CustomCNN(num_classes)),
    ]

    for mname, loader_fn in candidates:
        ckpt = os.path.join(save_dir, f"{mname}_best.pt")
        if not os.path.exists(ckpt):
            continue

        print(f"\n{'='*55}")
        print(f"Deployment Analysis: {mname}")
        print(f"{'='*55}")

        m = loader_fn()
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m = m.to(device)
        m.eval()

        # distribution shift
        run_distribution_shift(
            m, cfg["data"]["val_dir"], device, num_classes,
            log_dir=log_dir, model_name=mname
        )

        # energy
        run_energy_analysis(m, mname,
                            image_size=cfg["data"]["image_size"],
                            log_dir=log_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_deployment_analysis(args.config)
