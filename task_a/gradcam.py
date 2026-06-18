import os
import sys
import argparse
import yaml
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.custom_cnn import CustomCNN
from task_a.transfer_learning import get_model, SUPPORTED_MODELS


def get_target_layer(model, model_name):
    """
    Returns the correct conv layer for Grad-CAM per model type.
    Uses if/elif so we only access the attribute that belongs to THIS model.
    """
    if model_name == "custom":
        return model.block5
    elif model_name == "resnet50":
        return model.layer4[-1]
    elif model_name in ("efficientnet_b3", "vgg16", "mobilenet_v2"):
        return model.features[-1]
    elif model_name == "densenet121":
        return model.features.denseblock4
    else:
        raise ValueError(f"No target layer mapping for model '{model_name}'")


def run_gradcam(model, target_layer, input_tensor, class_idx):
    """
    Pure-PyTorch Grad-CAM — no extra library needed.
    Returns a numpy heatmap (H, W) in [0, 1].
    """
    gradients = []
    activations = []

    def forward_hook(module, inp, out):
        activations.append(out.detach())

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    model.zero_grad()
    output = model(input_tensor)

    # backprop w.r.t. the target class
    one_hot = torch.zeros_like(output)
    one_hot[0, class_idx] = 1.0
    output.backward(gradient=one_hot)

    fh.remove()
    bh.remove()

    grads = gradients[0]          # (1, C, H, W)
    acts  = activations[0]        # (1, C, H, W)

    # global average pool the gradients
    weights = grads.mean(dim=(2, 3), keepdim=True)   # (1, C, 1, 1)
    cam = (weights * acts).sum(dim=1, keepdim=True)   # (1, 1, H, W)
    cam = torch.relu(cam)

    # normalise to [0, 1]
    cam = cam.squeeze().cpu().numpy()
    cam_min, cam_max = cam.min(), cam.max()
    if cam_max - cam_min > 1e-8:
        cam = (cam - cam_min) / (cam_max - cam_min)

    return cam


def overlay_cam(image_np, cam, alpha=0.5):
    """Overlay CAM heatmap on the original image"""
    import cv2
    cam_resized = cv2.resize(cam, (image_np.shape[1], image_np.shape[0]))
    heatmap     = cv2.applyColorMap(
        (cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay     = alpha * heatmap + (1 - alpha) * image_np
    return np.clip(overlay, 0, 1)


def visualize_gradcam_batch(model, model_name, val_dir,
                             class_names, device, n_images=6,
                             save_path="logs/gradcam.png",
                             image_size=224):
    """Generate Grad-CAM for n_images from the val set"""
    from torchvision import datasets

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    inv_normalize = transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std =[1/0.229,      1/0.224,      1/0.225],
    )

    ds      = datasets.ImageFolder(val_dir, transform=tf)
    target_layer = get_target_layer(model, model_name)

    model.eval()
    fig, axes = plt.subplots(n_images, 2, figsize=(8, n_images * 3))

    shown = 0
    for idx in range(len(ds)):
        if shown >= n_images:
            break
        img_tensor, true_label = ds[idx]
        inp = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(inp)
        pred_label = logits.argmax(1).item()

        # grad-cam needs gradients
        inp.requires_grad_(True)
        cam = run_gradcam(model, target_layer, inp, pred_label)

        # original image for display
        orig_np = inv_normalize(img_tensor).permute(1, 2, 0).clamp(0, 1).numpy()
        overlay = overlay_cam(orig_np, cam)

        axes[shown, 0].imshow(orig_np)
        axes[shown, 0].set_title(
            f"True: {class_names[true_label][:20]}", fontsize=8
        )
        axes[shown, 0].axis("off")

        axes[shown, 1].imshow(overlay)
        axes[shown, 1].set_title(
            f"Pred: {class_names[pred_label][:20]}", fontsize=8
        )
        axes[shown, 1].axis("off")
        shown += 1

    plt.suptitle(f"Grad-CAM — {model_name}", fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved Grad-CAM figure → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default="custom")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    import yaml
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]

    if args.model == "custom":
        model = CustomCNN(num_classes)
    else:
        model = get_model(args.model, num_classes)

    ckpt = os.path.join(cfg["paths"]["save_dir"], f"{args.model}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)

    # need class_names — get from folder structure
    from torchvision import datasets
    ds = datasets.ImageFolder(cfg["data"]["val_dir"])
    class_names = ds.classes

    visualize_gradcam_batch(
        model       = model,
        model_name  = args.model,
        val_dir     = cfg["data"]["val_dir"],
        class_names = class_names,
        device      = device,
        n_images    = 8,
        save_path   = os.path.join(cfg["paths"]["log_dir"], f"{args.model}_gradcam.png"),
        image_size  = cfg["data"]["image_size"],
    )
