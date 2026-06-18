"""
Task C — Attention Rollout and entropy visualisation for ViT models.
Generates attention maps at layers 4, 8, 12 for 20 test images.
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import timm
from torchvision import datasets, transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_c.vit_timm import get_vit_model, VIT_MODELS
from utils.plotting import plot_attention_entropy


def get_attention_maps_vit(model, img_tensor, device):
    """
    Hooks into each TransformerBlock to capture attention weights.
    Returns list of (num_heads, seq_len, seq_len) attention maps per layer.
    Works for timm ViT and DeiT models.
    """
    attention_maps = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # timm attention modules expose attn_weights via forward
            # We capture the attention weights through the QKV computation
            pass
        return hook

    # timm ViT stores blocks in model.blocks
    hooks = []
    captured = []

    def attn_hook(module, inp, out):
        # 'out' from Attention module is (x, attn_weights) or just x
        # timm returns only x; we need to recompute attn weights
        captured.append(None)   # placeholder

    # Alternative: use register_forward_hook on the attn sub-module
    # and recompute attention from QKV
    if not hasattr(model, "blocks"):
        print("Model does not have .blocks attribute — skipping attention viz")
        return []

    qkv_outputs = []

    def qkv_hook(module, inp, out):
        qkv_outputs.append(out.detach().cpu())

    for block in model.blocks:
        if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
            hooks.append(block.attn.qkv.register_forward_hook(qkv_hook))

    model.eval()
    with torch.no_grad():
        _ = model(img_tensor.to(device))

    for h in hooks:
        h.remove()

    # convert QKV outputs to attention maps
    attn_maps = []
    for qkv in qkv_outputs:
        # qkv shape: (B, seq_len, 3 * num_heads * head_dim)
        B, N, _ = qkv.shape
        if not hasattr(model, "blocks"):
            continue
        # get num_heads from model config
        num_heads = model.blocks[0].attn.num_heads
        head_dim  = qkv.shape[-1] // (3 * num_heads)
        qkv_r = qkv.reshape(B, N, 3, num_heads, head_dim)
        qkv_r = qkv_r.permute(2, 0, 3, 1, 4)   # (3, B, H, N, D)
        q, k, v = qkv_r[0], qkv_r[1], qkv_r[2]
        scale    = head_dim ** -0.5
        attn     = (q @ k.transpose(-2, -1)) * scale
        attn     = attn.softmax(dim=-1)          # (B, H, N, N)
        attn_maps.append(attn[0])                # take first image: (H, N, N)

    return attn_maps


def attention_rollout(attn_maps):
    """
    Attention Rollout: propagate attention through layers.
    attn_maps: list of (num_heads, seq_len, seq_len) tensors
    Returns: (seq_len, seq_len) rollout map
    """
    result = torch.eye(attn_maps[0].shape[-1])

    for attn in attn_maps:
        # average over heads
        attn_avg = attn.mean(dim=0)   # (seq_len, seq_len)
        # add residual connection
        attn_avg = attn_avg + torch.eye(attn_avg.shape[-1])
        attn_avg = attn_avg / attn_avg.sum(dim=-1, keepdim=True)
        result   = attn_avg @ result

    # CLS token attends to patches
    mask = result[0, 1:]   # skip CLS token itself
    width = int(mask.shape[0] ** 0.5)
    mask  = mask.reshape(width, width).numpy()
    # normalise
    mask  = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    return mask


def compute_attention_entropy(attn_maps):
    """
    Entropy = -sum(p * log(p)) per head per layer.
    Returns list of mean entropy per layer.
    """
    entropies = []
    for attn in attn_maps:
        # attn: (num_heads, seq_len, seq_len)
        p       = attn.float() + 1e-10
        entropy = -(p * p.log()).sum(dim=-1).mean().item()
        entropies.append(entropy)
    return entropies


def visualize_attention(model_name, config_path="configs/config.yaml",
                        n_images=8, target_layers=(4, 8, 12)):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    image_size  = cfg["data"]["image_size"]
    save_dir    = cfg["paths"]["log_dir"]
    os.makedirs(save_dir, exist_ok=True)

    # load model
    model = get_vit_model(model_name, num_classes, pretrained=False)
    ckpt  = os.path.join(cfg["paths"]["save_dir"], f"{model_name}_best.pt")
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found: {ckpt}")
        return

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    # skip hybrid (no standard blocks)
    if model_name == "hybrid":
        print("Attention rollout not available for Hybrid CNN-ViT model.")
        return

    # data
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    inv_norm  = transforms.Normalize(
        [-0.485/0.229, -0.456/0.224, -0.406/0.225],
        [1/0.229, 1/0.224, 1/0.225]
    )
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    ds = datasets.ImageFolder(cfg["data"]["val_dir"], transform=tf)

    all_entropies = None
    shown = 0

    fig, axes = plt.subplots(n_images, len(target_layers) + 1,
                              figsize=(4 * (len(target_layers) + 1), 4 * n_images))

    for img_idx in range(len(ds)):
        if shown >= n_images:
            break

        img_tensor, true_label = ds[img_idx]
        inp = img_tensor.unsqueeze(0).to(device)

        attn_maps = get_attention_maps_vit(model, inp, device)
        if not attn_maps:
            print("Could not extract attention maps from this model.")
            break

        entropies = compute_attention_entropy(attn_maps)
        if all_entropies is None:
            all_entropies = entropies
        else:
            all_entropies = [a + b for a, b in zip(all_entropies, entropies)]

        # original image
        orig_np = inv_norm(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        axes[shown, 0].imshow(orig_np)
        axes[shown, 0].set_title(f"True: {ds.classes[true_label][:15]}", fontsize=7)
        axes[shown, 0].axis("off")

        # attention rollout at requested layers
        for col_idx, layer_idx in enumerate(target_layers):
            if layer_idx <= len(attn_maps):
                rollout = attention_rollout(attn_maps[:layer_idx])
                import cv2
                rollout_resized = cv2.resize(rollout, (image_size, image_size))
                axes[shown, col_idx + 1].imshow(orig_np, alpha=0.5)
                axes[shown, col_idx + 1].imshow(rollout_resized,
                                                  cmap="jet", alpha=0.5)
                axes[shown, col_idx + 1].set_title(f"Layer {layer_idx}", fontsize=7)
                axes[shown, col_idx + 1].axis("off")

        shown += 1

    plt.suptitle(f"Attention Rollout — {model_name}", fontsize=12)
    plt.tight_layout()
    attn_path = os.path.join(save_dir, f"{model_name}_attention_rollout.png")
    plt.savefig(attn_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved attention rollout → {attn_path}")

    # save entropy plot
    if all_entropies:
        mean_entropies = [e / n_images for e in all_entropies]
        plot_attention_entropy(
            mean_entropies,
            save_path=os.path.join(save_dir, f"{model_name}_entropy.png")
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="vit_b16",
                        choices=list(VIT_MODELS.keys()))
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--n",      type=int, default=8)
    args = parser.parse_args()
    visualize_attention(args.model, args.config, n_images=args.n)
