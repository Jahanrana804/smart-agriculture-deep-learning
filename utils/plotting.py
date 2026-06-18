import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # no display needed — saves to file
import seaborn as sns


def plot_training_curves(train_losses, val_accs, save_path="logs/training_curve.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(train_losses, color="steelblue")
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(val_accs, color="darkorange")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved training curves → {save_path}")


def plot_confusion_matrix(cm, class_names, save_path="logs/confusion_matrix.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(22, 20))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 7}
    )
    plt.title("Confusion Matrix", fontsize=16)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=90, fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved confusion matrix → {save_path}")


def plot_gan_losses(g_losses, d_losses, save_path="logs/gan_losses.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(g_losses, label="Generator",     color="steelblue")
    plt.plot(d_losses, label="Discriminator", color="darkorange")
    plt.title("GAN Training Losses")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved GAN losses → {save_path}")


def plot_attention_entropy(entropy_per_layer, save_path="logs/attention_entropy.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 4))
    layers = list(range(1, len(entropy_per_layer) + 1))
    plt.plot(layers, entropy_per_layer, marker="o", color="steelblue")
    plt.title("Attention Entropy Per Layer")
    plt.xlabel("Layer")
    plt.ylabel("Mean Entropy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved attention entropy → {save_path}")


def plot_reliability_diagram(confidences, accuracies, save_path="logs/reliability.png"):
    """For Task D calibration analysis"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    plt.plot(confidences, accuracies, "s-", color="steelblue", label="Model")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved reliability diagram → {save_path}")
