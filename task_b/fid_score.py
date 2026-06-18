"""
Task B — FID and Inception Score computation.
Computes FID between real images and GAN/DDPM generated images.
"""
import os
import sys
import argparse
import yaml
import torch
import numpy as np
from torchvision import datasets, transforms, models
from torchvision.utils import save_image
from torch.utils.data import DataLoader, TensorDataset
from scipy import linalg

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── InceptionV3 Feature Extractor ─────────────────────────────────────────────
def get_inception_model(device):
    """Load InceptionV3, strip classification head, keep pool features"""
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    )
    inception.fc = torch.nn.Identity()   # output pool features (2048-d)
    inception.eval().to(device)
    return inception


@torch.no_grad()
def get_activations(images_tensor, model, batch_size=50, device="cuda"):
    """
    images_tensor: (N, 3, H, W) in [0, 1]
    Returns: (N, 2048) numpy array of InceptionV3 features
    """
    model.eval()
    # Inception expects 299×299
    resize = transforms.Resize((299, 299), antialias=True)
    images_tensor = resize(images_tensor)

    all_acts = []
    for start in range(0, len(images_tensor), batch_size):
        batch = images_tensor[start : start + batch_size].to(device)
        acts  = model(batch)
        all_acts.append(acts.cpu().numpy())

    return np.concatenate(all_acts, axis=0)


def calculate_fid(real_acts, fake_acts):
    """
    Frechet Inception Distance between real and fake activation sets.
    Lower is better.
    """
    mu1, sigma1 = real_acts.mean(axis=0), np.cov(real_acts, rowvar=False)
    mu2, sigma2 = fake_acts.mean(axis=0), np.cov(fake_acts, rowvar=False)

    diff = mu1 - mu2
    # matrix square root of covariance product
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return float(fid)


@torch.no_grad()
def calculate_inception_score(images_tensor, device="cuda", batch_size=50, splits=10):
    """
    Inception Score: measures quality and diversity of generated images.
    Higher is better.
    """
    # Use inception for class predictions
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False
    ).to(device)
    inception.eval()

    resize = transforms.Resize((299, 299), antialias=True)
    images_tensor = resize(images_tensor)

    preds_list = []
    for start in range(0, len(images_tensor), batch_size):
        batch = images_tensor[start : start + batch_size].to(device)
        preds = torch.softmax(inception(batch), dim=1).cpu().numpy()
        preds_list.append(preds)

    preds = np.concatenate(preds_list, axis=0)   # (N, 1000)

    # compute IS over splits
    split_scores = []
    n = len(preds)
    for k in range(splits):
        part = preds[k * (n // splits) : (k + 1) * (n // splits)]
        py   = part.mean(axis=0, keepdims=True)   # marginal
        kl   = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl   = kl.sum(axis=1).mean()
        split_scores.append(np.exp(kl))

    return float(np.mean(split_scores)), float(np.std(split_scores))


def load_real_images(data_dir, n_images=2000, img_size=64):
    """Load n_images real images as a tensor in [0, 1]"""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
    ])
    ds     = datasets.ImageFolder(data_dir, transform=tf)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    all_imgs = []
    total    = 0
    for imgs, _ in loader:
        all_imgs.append(imgs)
        total += len(imgs)
        if total >= n_images:
            break

    return torch.cat(all_imgs)[:n_images]


def generate_dcgan_images(generator_path, n_images=2000,
                           latent_dim=100, img_size=64, device="cuda"):
    """Load saved DCGAN generator and generate n_images"""
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from task_b.dcgan import Generator

    G = Generator(latent_dim, img_size).to(device)
    G.load_state_dict(torch.load(generator_path, map_location=device))
    G.eval()

    all_imgs = []
    bs       = 64
    with torch.no_grad():
        for _ in range(0, n_images, bs):
            z     = torch.randn(min(bs, n_images - len(all_imgs) * bs),
                                latent_dim, device=device)
            imgs  = (G(z) * 0.5 + 0.5).cpu()
            all_imgs.append(imgs)

    return torch.cat(all_imgs)[:n_images]


def generate_ddpm_images(unet_path, n_images=500,
                          img_size=64, timesteps=1000, device="cuda"):
    """Load saved DDPM U-Net and generate n_images (slow — use small n)"""
    from task_b.ddpm import UNet, DDPM

    model = UNet(img_size=img_size).to(device)
    model.load_state_dict(torch.load(unet_path, map_location=device))

    ddpm    = DDPM(timesteps=timesteps, device=device)
    samples = []
    bs      = 16
    with torch.no_grad():
        for start in range(0, n_images, bs):
            batch_n = min(bs, n_images - start)
            imgs    = ddpm.sample(model, n_samples=batch_n,
                                  img_size=img_size) * 0.5 + 0.5
            samples.append(imgs.cpu())

    return torch.cat(samples)[:n_images]


def compute_all_metrics(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device   = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = cfg["gan"]["image_size"]
    save_dir = cfg["paths"]["save_dir"]

    inception = get_inception_model(device)

    print("\nLoading real images...")
    real_imgs = load_real_images(cfg["data"]["train_dir"],
                                  n_images=2000, img_size=img_size)
    real_acts = get_activations(real_imgs, inception, device=device)
    print(f"Real image activations: {real_acts.shape}")

    results = {}

    # ── DCGAN ─────────────────────────────────────────────────────
    dcgan_path = os.path.join(save_dir, "dcgan_generator.pt")
    if os.path.exists(dcgan_path):
        print("\nEvaluating DCGAN...")
        fake_imgs = generate_dcgan_images(dcgan_path, n_images=2000,
                                           img_size=img_size, device=device)
        fake_acts = get_activations(fake_imgs, inception, device=device)
        fid       = calculate_fid(real_acts, fake_acts)
        is_mean, is_std = calculate_inception_score(fake_imgs, device=device)

        print(f"  DCGAN FID: {fid:.2f}")
        print(f"  DCGAN IS:  {is_mean:.2f} ± {is_std:.2f}")
        results["dcgan"] = {"fid": fid, "is_mean": is_mean, "is_std": is_std}

        # save 8×8 sample grid
        grid_imgs = fake_imgs[:64]
        save_image(grid_imgs,
                   os.path.join(cfg["paths"]["samples_dir"], "dcgan_eval_grid.png"),
                   nrow=8)
    else:
        print(f"DCGAN checkpoint not found: {dcgan_path}")

    # ── DDPM ──────────────────────────────────────────────────────
    ddpm_path = os.path.join(save_dir, "ddpm_unet.pt")
    if os.path.exists(ddpm_path):
        print("\nEvaluating DDPM (generating 200 samples — this takes time)...")
        fake_imgs_ddpm = generate_ddpm_images(ddpm_path, n_images=200,
                                               img_size=img_size, device=device)
        # use subset of real for fair comparison
        real_acts_small = get_activations(real_imgs[:200], inception, device=device)
        fake_acts_ddpm  = get_activations(fake_imgs_ddpm, inception, device=device)
        fid_ddpm        = calculate_fid(real_acts_small, fake_acts_ddpm)
        is_mean_d, is_std_d = calculate_inception_score(fake_imgs_ddpm, device=device)

        print(f"  DDPM FID: {fid_ddpm:.2f}")
        print(f"  DDPM IS:  {is_mean_d:.2f} ± {is_std_d:.2f}")
        results["ddpm"] = {"fid": fid_ddpm, "is_mean": is_mean_d, "is_std": is_std_d}

        save_image(fake_imgs_ddpm[:16],
                   os.path.join(cfg["paths"]["samples_dir"], "ddpm_eval_grid.png"),
                   nrow=4)
    else:
        print(f"DDPM checkpoint not found: {ddpm_path}")

    # print summary table
    print("\n" + "="*45)
    print(f"{'Model':<10} {'FID':>8} {'IS Mean':>10} {'IS Std':>8}")
    print("-"*45)
    for name, vals in results.items():
        print(f"{name:<10} {vals['fid']:>8.2f} "
              f"{vals['is_mean']:>10.2f} {vals['is_std']:>8.2f}")
    print("="*45)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    compute_all_metrics(args.config)
