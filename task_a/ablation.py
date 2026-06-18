"""
Ablation study runner — Task A.3
Systematically varies: augmentation intensity, scheduler type,
and number of frozen layers, then logs results to a CSV table.
"""
import os
import sys
import csv
import yaml
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model
from task_a.train import train_one_epoch
from utils.metrics import evaluate_model


def run_single_experiment(model, train_dl, val_dl, device, epochs, lr,
                           scheduler_type="cosine", weight_decay=1e-4):
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=7, gamma=0.1
        )

    criterion = nn.CrossEntropyLoss()
    best_val  = 0.0

    for epoch in range(epochs):
        train_one_epoch(model, train_dl, optimizer, criterion, device)
        metrics = evaluate_model(model, val_dl, device,
                                 num_classes=len(train_dl.dataset.classes)
                                 if hasattr(train_dl.dataset, "classes")
                                 else 38)
        if metrics["accuracy"] > best_val:
            best_val = metrics["accuracy"]
        scheduler.step()

    return best_val


def run_ablation(config_path="configs/config.yaml", ablation_epochs=5):
    """
    ablation_epochs: use fewer epochs for quick ablation (full training later)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(cfg["paths"]["log_dir"], exist_ok=True)
    results = []

    # ── Ablation 1: Augmentation intensity ────────────────────────
    print("\n[Ablation 1] Augmentation intensity")
    for intensity in ["light", "medium", "heavy"]:
        train_dl, val_dl, class_names = get_dataloaders(
            cfg["data"]["train_dir"], cfg["data"]["val_dir"],
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"]["num_workers"],
            aug_intensity=intensity,
            image_size=cfg["data"]["image_size"],
        )
        model = CustomCNN(len(class_names)).to(device)
        acc = run_single_experiment(
            model, train_dl, val_dl, device,
            epochs=ablation_epochs, lr=cfg["train"]["lr"],
            scheduler_type="cosine"
        )
        print(f"  {intensity:6s} aug → best val acc: {acc:.4f}")
        results.append({
            "ablation":   "augmentation",
            "variant":    intensity,
            "model":      "custom_cnn",
            "scheduler":  "cosine",
            "val_acc":    round(acc, 4),
        })

    # ── Ablation 2: Scheduler type ────────────────────────────────
    print("\n[Ablation 2] Scheduler type")
    train_dl, val_dl, class_names = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        aug_intensity="medium",
        image_size=cfg["data"]["image_size"],
    )
    for sched in ["cosine", "step"]:
        model = CustomCNN(len(class_names)).to(device)
        acc = run_single_experiment(
            model, train_dl, val_dl, device,
            epochs=ablation_epochs, lr=cfg["train"]["lr"],
            scheduler_type=sched
        )
        print(f"  {sched:6s} scheduler → best val acc: {acc:.4f}")
        results.append({
            "ablation":   "scheduler",
            "variant":    sched,
            "model":      "custom_cnn",
            "scheduler":  sched,
            "val_acc":    round(acc, 4),
        })

    # ── Ablation 3: Frozen layers (transfer learning) ─────────────
    print("\n[Ablation 3] Frozen layer depth (ResNet-50)")

    def get_resnet_with_n_unfrozen_blocks(num_classes, unfrozen):
        """
        unfrozen=0 → only head trainable
        unfrozen=1 → layer4 + head
        unfrozen=2 → layer3 + layer4 + head  (spec default)
        unfrozen=3 → layer2 + layer3 + layer4 + head
        """
        from torchvision import models
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for p in m.parameters():
            p.requires_grad = False
        blocks = [m.layer4, m.layer3, m.layer2, m.layer1]
        for i in range(min(unfrozen, len(blocks))):
            for p in blocks[i].parameters():
                p.requires_grad = True
        m.fc = nn.Linear(2048, num_classes)
        return m

    for n_unfrozen in [0, 1, 2, 3]:
        model = get_resnet_with_n_unfrozen_blocks(
            len(class_names), n_unfrozen
        ).to(device)
        acc = run_single_experiment(
            model, train_dl, val_dl, device,
            epochs=ablation_epochs, lr=cfg["train"]["lr"],
            scheduler_type="cosine"
        )
        print(f"  {n_unfrozen} unfrozen blocks → best val acc: {acc:.4f}")
        results.append({
            "ablation":   "frozen_layers",
            "variant":    f"{n_unfrozen}_unfrozen",
            "model":      "resnet50",
            "scheduler":  "cosine",
            "val_acc":    round(acc, 4),
        })

    # ── Save results to CSV ───────────────────────────────────────
    csv_path = os.path.join(cfg["paths"]["log_dir"], "ablation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAblation results saved → {csv_path}")
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Epochs per ablation run (keep low for speed)")
    args = parser.parse_args()
    run_ablation(args.config, args.epochs)
