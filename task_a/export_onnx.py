import os
import sys
import argparse
import yaml
import torch
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model, SUPPORTED_MODELS


def export_to_onnx(model, save_path, image_size=224):
    """Export PyTorch model to ONNX using legacy API (no onnxscript needed)"""
    model.eval().cpu()
    dummy = torch.randn(1, 3, image_size, image_size)
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            save_path,
            input_names         = ["input"],
            output_names        = ["output"],
            dynamic_axes        = {"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version       = 11,
            do_constant_folding = True,
            export_params       = True,
        )
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f"Exported ONNX model -> {save_path}  ({size_mb:.1f} MB)")
    return save_path


def apply_int8_quantization(model):
    """
    Post-training INT8 quantization on CPU.
    Simulates what would run on a Raspberry Pi / Jetson.
    """
    model.eval().cpu()
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear, torch.nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized


def benchmark_speed(model, device="cpu", image_size=224, n_runs=200, label=""):
    """Returns mean latency in ms over n_runs single-image passes"""
    model = model.to(device)
    model.eval()
    dummy = torch.randn(1, 3, image_size, image_size).to(device)

    # warmup
    with torch.no_grad():
        for _ in range(20):
            _ = model(dummy)

    # timed
    times = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    std_ms  = float(np.std(times))
    print(f"  [{label or device}] {mean_ms:.2f} ± {std_ms:.2f} ms/image")
    return mean_ms


def run_edge_analysis(model_name="custom", config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    num_classes = cfg["data"]["num_classes"]
    image_size  = cfg["data"]["image_size"]
    save_dir    = cfg["paths"]["save_dir"]

    if model_name == "custom":
        model = CustomCNN(num_classes)
    else:
        model = get_model(model_name, num_classes)

    ckpt = os.path.join(save_dir, f"{model_name}_best.pt")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}  — train first.")
        return

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))

    print(f"\n{'='*55}")
    print(f"Edge Deployment Analysis: {model_name}")
    print(f"{'='*55}")

    # 1. Export ONNX
    onnx_path = os.path.join(save_dir, f"{model_name}.onnx")
    export_to_onnx(model, onnx_path, image_size)

    # 2. GPU latency (if available)
    if torch.cuda.is_available():
        benchmark_speed(model, "cuda", image_size, label="GPU (RTX 5070)")

    # 3. CPU latency (simulates edge device)
    benchmark_speed(model, "cpu", image_size, label="CPU (simulated edge)")

    # 4. INT8 quantization
    print("\n  Applying INT8 post-training quantization...")
    quantized_model = apply_int8_quantization(model)

    # model size comparison
    fp32_size = sum(
        p.numel() * p.element_size() for p in model.parameters()
    ) / (1024 * 1024)

    print(f"  FP32 model size: {fp32_size:.1f} MB")
    print(f"  INT8 model size: ~{fp32_size / 4:.1f} MB (approx 4x reduction)")

    benchmark_speed(quantized_model, "cpu", image_size, label="CPU INT8 quantized")

    # 5. Raspberry Pi / Jetson suitability
    print("\n  Edge Device Suitability:")
    print("  ┌──────────────────────────────────────────────────┐")
    print("  │ Device          │ Suitable │ Notes               │")
    print("  ├──────────────────────────────────────────────────┤")
    print("  │ Raspberry Pi 4  │ Maybe    │ Use quantized model │")
    print("  │ Jetson Nano     │ Yes      │ GPU-accelerated     │")
    print("  │ Jetson Xavier   │ Yes      │ FP16 TensorRT       │")
    print("  │ Smartphone      │ Yes      │ ONNX Runtime        │")
    print("  └──────────────────────────────────────────────────┘")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="custom")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_edge_analysis(args.model, args.config)
