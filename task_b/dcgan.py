"""
Task B — DCGAN
Generates synthetic images for the minority class in the dataset.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.plotting import plot_gan_losses


# ── Generator ─────────────────────────────────────────────────────────────────
class Generator(nn.Module):
    """
    Noise z (latent_dim,) → RGB image (3, img_size, img_size)
    Uses transposed convolutions to upsample from 4×4 to img_size×img_size.
    """
    def __init__(self, latent_dim=100, img_size=64, channels=3):
        super().__init__()
        self.init_size = img_size // 16   # = 4 for img_size=64

        self.fc = nn.Linear(latent_dim, 512 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.BatchNorm2d(512),
            # 4×4 → 8×8
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8×8 → 16×16
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 16×16 → 32×32
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 32×32 → 64×64
            nn.ConvTranspose2d(64, channels, 4, stride=2, padding=1, bias=False),
            nn.Tanh(),   # output in [-1, 1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        return self.net(x)


# ── Discriminator ─────────────────────────────────────────────────────────────
class Discriminator(nn.Module):
    """
    RGB image (3, img_size, img_size) → scalar probability in [0, 1]
    """
    def __init__(self, img_size=64, channels=3):
        super().__init__()

        def disc_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *disc_block(channels, 64,  bn=False),  # 64→32
            *disc_block(64,  128),                  # 32→16
            *disc_block(128, 256),                  # 16→8
            *disc_block(256, 512),                  # 8→4
            nn.Conv2d(512, 1, 4, stride=1, padding=0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def weights_init(m):
    """DCGAN weight initialisation from the original paper"""
    classname = m.__class__.__name__
    if "Conv" in classname:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif "BatchNorm" in classname:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def get_single_class_loader(data_dir, target_class_idx, img_size, batch_size, num_workers=4):
    """Returns a dataloader with only images from one class (minority class)"""
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    full_ds = datasets.ImageFolder(data_dir, transform=tf)

    # filter to target class
    indices = [i for i, (_, lbl) in enumerate(full_ds.samples) if lbl == target_class_idx]
    subset  = torch.utils.data.Subset(full_ds, indices)

    loader = DataLoader(subset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers, pin_memory=True)
    print(f"DCGAN target class: {full_ds.classes[target_class_idx]} "
          f"({len(subset)} images)")
    return loader, full_ds.classes[target_class_idx]


def train_dcgan(config_path="configs/config.yaml", target_class_idx=0):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    g_cfg    = cfg["gan"]
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = g_cfg["image_size"]
    latent   = g_cfg["latent_dim"]
    epochs   = g_cfg["epochs"]
    lr       = g_cfg["lr"]
    bs       = g_cfg["batch_size"]

    os.makedirs(cfg["paths"]["save_dir"],    exist_ok=True)
    os.makedirs(cfg["paths"]["samples_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"],     exist_ok=True)

    dataloader, class_name = get_single_class_loader(
        cfg["data"]["train_dir"], target_class_idx, img_size, bs
    )

    G = Generator(latent, img_size).to(device)
    D = Discriminator(img_size).to(device)
    G.apply(weights_init)
    D.apply(weights_init)

    opt_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    fixed_z   = torch.randn(64, latent, device=device)
    g_losses, d_losses = [], []

    print(f"\nTraining DCGAN on class: {class_name}")
    print(f"Epochs: {epochs} | Image size: {img_size} | Device: {device}\n")

    for epoch in range(epochs):
        epoch_g, epoch_d = 0.0, 0.0

        for real_imgs, _ in dataloader:
            real_imgs = real_imgs.to(device)
            bs_cur    = real_imgs.size(0)

            real_labels = torch.ones(bs_cur,  device=device) * 0.9  # label smoothing
            fake_labels = torch.zeros(bs_cur, device=device)

            # ── Train Discriminator ──────────────────────
            z    = torch.randn(bs_cur, latent, device=device)
            fake = G(z).detach()

            loss_D_real = criterion(D(real_imgs), real_labels)
            loss_D_fake = criterion(D(fake),      fake_labels)
            loss_D      = (loss_D_real + loss_D_fake) / 2

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()

            # ── Train Generator ──────────────────────────
            z    = torch.randn(bs_cur, latent, device=device)
            fake = G(z)
            # generator wants D to output 1 (real) for fakes
            loss_G = criterion(D(fake), torch.ones(bs_cur, device=device))

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            epoch_g += loss_G.item()
            epoch_d += loss_D.item()

        g_losses.append(epoch_g / len(dataloader))
        d_losses.append(epoch_d / len(dataloader))

        print(f"Epoch [{epoch+1:>3}/{epochs}] "
              f"D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f}")

        # save sample grid every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                samples = G(fixed_z) * 0.5 + 0.5   # rescale to [0, 1]
            save_image(
                samples,
                os.path.join(cfg["paths"]["samples_dir"],
                             f"dcgan_epoch_{epoch+1}.png"),
                nrow=8,
            )

    # save model
    torch.save(G.state_dict(),
               os.path.join(cfg["paths"]["save_dir"], "dcgan_generator.pt"))
    torch.save(D.state_dict(),
               os.path.join(cfg["paths"]["save_dir"], "dcgan_discriminator.pt"))

    # loss curves
    plot_gan_losses(
        g_losses, d_losses,
        save_path=os.path.join(cfg["paths"]["log_dir"], "dcgan_losses.png")
    )

    print(f"\nDCGAN training complete. Generator saved.")
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/config.yaml")
    parser.add_argument("--target_class", type=int, default=0,
                        help="Index of the minority class to generate")
    args = parser.parse_args()
    train_dcgan(args.config, args.target_class)
