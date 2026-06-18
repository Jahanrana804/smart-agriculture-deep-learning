"""
Task B — Conditional GAN (cGAN)
Class-conditional image synthesis.
Generator accepts a class label embedding alongside the noise vector.
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


class ConditionalGenerator(nn.Module):
    """
    Input: noise z (latent_dim,) + class label → image (3, img_size, img_size)
    Label is converted to an embedding and concatenated with z before FC.
    """
    def __init__(self, num_classes, latent_dim=100, img_size=64, embed_dim=50):
        super().__init__()
        self.embed = nn.Embedding(num_classes, embed_dim)
        self.init_size = img_size // 16   # 4 for img_size=64

        self.fc = nn.Linear(latent_dim + embed_dim,
                             512 * self.init_size * self.init_size)

        self.net = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        label_embed = self.embed(labels)            # (B, embed_dim)
        x = torch.cat([z, label_embed], dim=1)     # (B, latent+embed)
        x = self.fc(x)
        x = x.view(x.size(0), 512, self.init_size, self.init_size)
        return self.net(x)


class ConditionalDiscriminator(nn.Module):
    """
    Input: image (3, img_size, img_size) + class label → real/fake probability
    Label is embedded and added as an extra channel (broadcast to spatial dims).
    """
    def __init__(self, num_classes, img_size=64, embed_dim=50):
        super().__init__()
        self.img_size = img_size
        self.embed    = nn.Embedding(num_classes, embed_dim)
        self.embed_fc = nn.Linear(embed_dim, img_size * img_size)

        def disc_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
            if bn:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        # input channels = 3 (image) + 1 (label map)
        self.net = nn.Sequential(
            *disc_block(4,   64,  bn=False),
            *disc_block(64,  128),
            *disc_block(128, 256),
            *disc_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, imgs, labels):
        # embed label → spatial map (B, 1, H, W)
        lbl_embed = self.embed(labels)                           # (B, embed_dim)
        lbl_map   = self.embed_fc(lbl_embed)                     # (B, H*W)
        lbl_map   = lbl_map.view(-1, 1, self.img_size, self.img_size)
        x = torch.cat([imgs, lbl_map], dim=1)                    # (B, 4, H, W)
        return self.net(x).view(-1)


def train_cgan(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    g_cfg       = cfg["gan"]
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    img_size    = g_cfg["image_size"]
    latent      = g_cfg["latent_dim"]
    epochs      = g_cfg["epochs"]
    lr          = g_cfg["lr"]
    bs          = g_cfg["batch_size"]
    num_classes = cfg["data"]["num_classes"]

    os.makedirs(cfg["paths"]["save_dir"],    exist_ok=True)
    os.makedirs(cfg["paths"]["samples_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"],     exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

    G = ConditionalGenerator(num_classes, latent, img_size).to(device)
    D = ConditionalDiscriminator(num_classes, img_size).to(device)

    opt_G     = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_D     = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))
    criterion = nn.BCELoss()

    # fixed noise + labels for consistent sample grids
    fixed_z      = torch.randn(num_classes * 4, latent, device=device)
    fixed_labels = torch.arange(num_classes, device=device).repeat(4)

    g_losses, d_losses = [], []
    print(f"\nTraining cGAN | Classes: {num_classes} | Device: {device}")

    for epoch in range(epochs):
        eg, ed = 0.0, 0.0

        for real_imgs, labels in dl:
            real_imgs = real_imgs.to(device)
            labels    = labels.to(device)
            bsz       = real_imgs.size(0)

            real_lbl = torch.ones(bsz,  device=device) * 0.9
            fake_lbl = torch.zeros(bsz, device=device)

            # ── Discriminator ────────────────────────
            z    = torch.randn(bsz, latent, device=device)
            fake = G(z, labels).detach()

            loss_D = (criterion(D(real_imgs, labels), real_lbl) +
                      criterion(D(fake,      labels), fake_lbl)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()

            # ── Generator ────────────────────────────
            z    = torch.randn(bsz, latent, device=device)
            fake = G(z, labels)
            loss_G = criterion(D(fake, labels),
                               torch.ones(bsz, device=device))
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()

            eg += loss_G.item()
            ed += loss_D.item()

        g_losses.append(eg / len(dl))
        d_losses.append(ed / len(dl))
        print(f"Epoch [{epoch+1:>3}/{epochs}]  "
              f"D: {d_losses[-1]:.4f} | G: {g_losses[-1]:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == 0:
            with torch.no_grad():
                samples = G(fixed_z, fixed_labels) * 0.5 + 0.5
            save_image(
                samples,
                os.path.join(cfg["paths"]["samples_dir"],
                             f"cgan_epoch_{epoch+1}.png"),
                nrow=num_classes,
            )

    torch.save(G.state_dict(),
               os.path.join(cfg["paths"]["save_dir"], "cgan_generator.pt"))

    plot_gan_losses(
        g_losses, d_losses,
        save_path=os.path.join(cfg["paths"]["log_dir"], "cgan_losses.png")
    )
    print("\ncGAN training complete.")
    return G


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_cgan(args.config)
