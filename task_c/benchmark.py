"""
Task C.4 — Comprehensive benchmark table.
Evaluates all Task A and Task C models and prints the filled table.
"""
import os
import sys
import argparse
import yaml
import torch
import time
import csv
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model as get_cnn_model
from task_c.vit_timm import get_vit_model, VIT_MODELS
from utils.metrics import evaluate_model


ALL_MODELS = {
    # (type, name_for_get_model)
    "Custom CNN":     ("cnn",  "custom"),
    "ResNet-50":      ("cnn",  "resnet50"),
    "EfficientNet-B3":("cnn",  "efficientnet_b3"),
    "MobileNetV2":    ("cnn",  "mobilenet_v2"),
    "ViT-B/16":       ("vit",  "vit_b16"),
    "DeiT-Small":     ("vit",  "deit_small"),
    "Swin-Tiny":      ("vit",  "swin_tiny"),
    "Hybrid CNN-ViT": ("vit",  "hybrid"),
}

EDGE_THRESHOLD_MS = 100   # <100ms = deployable on edge


def model_size_mb(model):
    total = sum(p.numel() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


def benchmark_latency_ms(model, device, image_size=224, n=100):
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)
    with torch.no_grad():
        for _ in range(10): _ = model(dummy)   # warmup
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n): _ = model(dummy)
    return (time.perf_counter() - start) / n * 1000


def get_flops_gflops(model, image_size=224):
    try:
        from thop import profile
        dummy = torch.randn(1, 3, image_size, image_size)
        flops, _ = profile(model.cpu(), inputs=(dummy,), verbose=False)
        return flops / 1e9
    except Exception:
        return None


def build_benchmark_table(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    image_size  = cfg["data"]["image_size"]
    save_dir    = cfg["paths"]["save_dir"]

    _, val_dl, class_names = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        image_size  = image_size,
    )

    rows = []

    for display_name, (mtype, mname) in ALL_MODELS.items():
        ckpt = os.path.join(save_dir, f"{mname}_best.pt")

        if not os.path.exists(ckpt):
            print(f"  [{display_name}] No checkpoint — skipping")
            rows.append({
                "Model": display_name,
                "Top-1 Acc": "—",
                "F1": "—",
                "Params (M)": "—",
                "GFLOPs": "—",
                "CPU Lat (ms)": "—",
                "Deploy?": "—",
            })
            continue

        print(f"  Evaluating: {display_name}")

        # load model
        if mtype == "cnn":
            if mname == "custom":
                model = CustomCNN(num_classes)
            else:
                model = get_cnn_model(mname, num_classes)
        else:
            model = get_vit_model(mname, num_classes, pretrained=False)

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)
        model.eval()

        # metrics
        metrics = evaluate_model(model, val_dl, device, num_classes)
        acc     = metrics["accuracy"]
        f1      = metrics["f1"]

        # params
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        # FLOPs
        gflops = get_flops_gflops(model, image_size)

        # CPU latency
        cpu_lat = benchmark_latency_ms(model.cpu(), "cpu", image_size)

        # deployable?
        deployable = "Yes" if cpu_lat < EDGE_THRESHOLD_MS else "No"

        rows.append({
            "Model":         display_name,
            "Top-1 Acc":     f"{acc:.4f}",
            "F1":            f"{f1:.4f}",
            "Params (M)":    f"{params_m:.1f}",
            "GFLOPs":        f"{gflops:.2f}" if gflops else "—",
            "CPU Lat (ms)":  f"{cpu_lat:.1f}",
            "Deploy?":       deployable,
        })

    # print table
    headers = ["Model", "Top-1 Acc", "F1", "Params (M)", "GFLOPs",
               "CPU Lat (ms)", "Deploy?"]
    col_w   = [max(len(h), max((len(str(r[h])) for r in rows), default=0))
               for h in headers]

    sep  = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"
    head = "|" + "|".join(f" {h:<{w}} " for h, w in zip(headers, col_w)) + "|"

    print("\n" + sep)
    print(head)
    print(sep)
    for row in rows:
        line = "|" + "|".join(
            f" {str(row[h]):<{w}} " for h, w in zip(headers, col_w)
        ) + "|"
        print(line)
    print(sep)

    # save to CSV
    csv_path = os.path.join(cfg["paths"]["log_dir"], "benchmark_table.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nBenchmark table saved → {csv_path}")

    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    build_benchmark_table(args.config)
