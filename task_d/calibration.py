"""
Task D — Expected Calibration Error (ECE) and reliability diagrams.
Measures whether model confidence matches actual accuracy.
"""
import os
import sys
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model as get_cnn_model
from task_c.vit_timm import get_vit_model


@torch.no_grad()
def get_confidences_and_labels(model, dataloader, device):
    """
    Returns:
        confidences  — max softmax probability for each sample
        correctness  — 1 if prediction correct, 0 otherwise
        all_probs    — full softmax distribution (N, C)
        all_preds    — predicted class indices (N,)
        all_labels   — true labels (N,)
    """
    model.eval()
    confs, correct, probs_list, preds_list, labels_list = [], [], [], [], []

    for imgs, labels in dataloader:
        imgs   = imgs.to(device)
        logits = model(imgs)
        probs  = torch.softmax(logits, dim=1).cpu()
        preds  = probs.argmax(1)

        confs.append(probs.max(1).values)
        correct.append((preds == labels).float())
        probs_list.append(probs)
        preds_list.append(preds)
        labels_list.append(labels)

    return (
        torch.cat(confs).numpy(),
        torch.cat(correct).numpy(),
        torch.cat(probs_list).numpy(),
        torch.cat(preds_list).numpy(),
        torch.cat(labels_list).numpy(),
    )


def compute_ece(confidences, correctness, n_bins=15):
    """
    Expected Calibration Error.
    Bins samples by confidence, measures |acc - conf| per bin.
    Lower ECE = better calibrated.
    """
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs  = []
    bin_confs = []
    bin_sizes = []

    for low, high in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            bin_accs.append(0)
            bin_confs.append((low + high) / 2)
            bin_sizes.append(0)
            continue

        bin_acc  = correctness[mask].mean()
        bin_conf = confidences[mask].mean()
        bin_size = mask.sum()

        ece += (bin_size / len(confidences)) * abs(bin_acc - bin_conf)
        bin_accs.append(bin_acc)
        bin_confs.append(bin_conf)
        bin_sizes.append(bin_size)

    return ece, np.array(bin_accs), np.array(bin_confs), np.array(bin_sizes)


def plot_reliability(bin_confs, bin_accs, ece, model_name,
                     save_path="logs/reliability.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.bar(bin_confs, bin_accs, width=1.0 / len(bin_confs),
           alpha=0.7, color="steelblue", label="Model accuracy")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Perfect calibration")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram — {model_name}\nECE = {ece:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram → {save_path}")


def run_calibration_analysis(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    save_dir    = cfg["paths"]["save_dir"]
    log_dir     = cfg["paths"]["log_dir"]
    os.makedirs(log_dir, exist_ok=True)

    _, val_dl, _ = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        batch_size  = cfg["data"]["batch_size"],
        num_workers = cfg["data"]["num_workers"],
        image_size  = cfg["data"]["image_size"],
    )

    models_to_eval = [
        ("resnet50",   lambda: get_cnn_model("resnet50", num_classes)),
        ("custom",     lambda: CustomCNN(num_classes)),
        ("vit_b16",    lambda: get_vit_model("vit_b16",    num_classes, pretrained=False)),
        ("deit_small", lambda: get_vit_model("deit_small", num_classes, pretrained=False)),
        ("swin_tiny",  lambda: get_vit_model("swin_tiny",  num_classes, pretrained=False)),
    ]

    results = {}

    print("\n" + "="*50)
    print(f"{'Model':<20} {'ECE':>8} {'Avg Conf':>10} {'Accuracy':>10}")
    print("-"*50)

    for mname, loader_fn in models_to_eval:
        ckpt = os.path.join(save_dir, f"{mname}_best.pt")
        if not os.path.exists(ckpt):
            continue

        try:
            m = loader_fn()
            m.load_state_dict(torch.load(ckpt, map_location=device))
            m = m.to(device)

            confs, correct, _, _, _ = get_confidences_and_labels(m, val_dl, device)
            ece, bin_accs, bin_confs, _ = compute_ece(confs, correct)

            avg_conf = confs.mean()
            accuracy = correct.mean()

            print(f"{mname:<20} {ece:>8.4f} {avg_conf:>10.4f} {accuracy:>10.4f}")

            plot_reliability(
                bin_confs, bin_accs, ece, mname,
                save_path=os.path.join(log_dir, f"{mname}_reliability.png")
            )

            results[mname] = {
                "ece":      float(ece),
                "avg_conf": float(avg_conf),
                "accuracy": float(accuracy),
                "overconfident": bool(avg_conf > accuracy + 0.05),
            }
        except Exception as e:
            print(f"  [{mname}] Error: {e}")

    print("="*50)

    # overconfidence summary
    print("\nOverconfidence analysis:")
    for name, vals in results.items():
        status = "OVERCONFIDENT" if vals["overconfident"] else "Well calibrated"
        print(f"  {name:<20} → {status} "
              f"(conf={vals['avg_conf']:.3f}, acc={vals['accuracy']:.3f})")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_calibration_analysis(args.config)
