import os
import sys
import argparse
import yaml
import torch
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model, SUPPORTED_MODELS
from utils.metrics import evaluate_model, count_parameters
from utils.plotting import plot_confusion_matrix


def benchmark_latency(model, device, image_size=224, n_runs=100):
    """Measure average inference time in ms — simulates edge deployment"""
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy)

    # timed runs
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(dummy)
    elapsed_ms = (time.perf_counter() - start) / n_runs * 1000

    print(f"  Latency on {device}: {elapsed_ms:.2f} ms/image")
    return elapsed_ms


def compute_flops(model, image_size=224):
    """Approximate FLOPs using thop if available, else skip"""
    try:
        from thop import profile
        dummy = torch.randn(1, 3, image_size, image_size)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        gflops = flops / 1e9
        print(f"  FLOPs: {gflops:.2f} GFLOPs")
        return gflops
    except ImportError:
        print("  FLOPs: install 'thop' for FLOPs counting  (pip install thop)")
        return None


def evaluate(model_name="custom", config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, val_dl, class_names = get_dataloaders(
        train_dir   = cfg["data"]["train_dir"],
        val_dir     = cfg["data"]["val_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        image_size  = cfg["data"]["image_size"],
    )
    num_classes = len(class_names)

    # load model
    if model_name == "custom":
        model = CustomCNN(num_classes)
    else:
        model = get_model(model_name, num_classes)

    ckpt_path = os.path.join(cfg["paths"]["save_dir"], f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}. Train first.")
        return

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"\n{'='*50}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*50}")

    # params
    count_parameters(model)

    # accuracy + f1 + confusion matrix
    metrics = evaluate_model(model, val_dl, device, num_classes)
    print(f"\n  Val Accuracy : {metrics['accuracy']:.4f}")
    print(f"  Val F1 Macro : {metrics['f1']:.4f}")

    # confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        save_path=os.path.join(cfg["paths"]["log_dir"], f"{model_name}_cm_eval.png")
    )

    # latency — GPU and CPU
    gpu_lat = benchmark_latency(model, "cuda", cfg["data"]["image_size"]) \
              if torch.cuda.is_available() else None
    cpu_model = model.to("cpu")
    cpu_lat = benchmark_latency(cpu_model, "cpu", cfg["data"]["image_size"])

    # FLOPs
    compute_flops(cpu_model, cfg["data"]["image_size"])

    # summary row for the benchmark table
    print(f"\n{'='*50}")
    print(f"BENCHMARK TABLE ROW [{model_name}]")
    print(f"  Acc:     {metrics['accuracy']:.4f}")
    print(f"  F1:      {metrics['f1']:.4f}")
    if gpu_lat: print(f"  GPU lat: {gpu_lat:.1f} ms")
    print(f"  CPU lat: {cpu_lat:.1f} ms")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="custom",
                        help=f"custom | {' | '.join(SUPPORTED_MODELS)}")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    evaluate(model_name=args.model, config_path=args.config)
