import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def get_accuracy(preds, labels):
    return (preds == labels).float().mean().item()


def get_f1(preds, labels):
    return f1_score(
        labels.cpu().numpy(),
        preds.cpu().numpy(),
        average="macro",
        zero_division=0
    )


def get_confusion_matrix(preds, labels, num_classes):
    return confusion_matrix(
        labels.cpu().numpy(),
        preds.cpu().numpy(),
        labels=list(range(num_classes))
    )


def evaluate_model(model, dataloader, device, num_classes):
    """
    Full evaluation pass.
    Returns dict with accuracy, f1, confusion_matrix, and per-class report.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs   = imgs.to(device)
            preds  = model(imgs).argmax(1).cpu()
            all_preds.append(preds)
            all_labels.append(labels)

    all_preds  = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    acc = get_accuracy(all_preds, all_labels)
    f1  = get_f1(all_preds, all_labels)
    cm  = get_confusion_matrix(all_preds, all_labels, num_classes)

    return {
        "accuracy":         acc,
        "f1":               f1,
        "confusion_matrix": cm,
        "preds":            all_preds,
        "labels":           all_labels,
    }


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")
    return total, trainable
