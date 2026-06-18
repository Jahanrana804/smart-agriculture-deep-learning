"""
Task B — DDPM (Denoising Diffusion Probabilistic Model)
T=1000 timesteps, U-Net backbone with sinusoidal time embeddings
and self-attention at 16×16 resolution as specified.
"""
import os
import sys
import argparse
import math
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def num_groups_for(channels, preferred=8):
    """Return largest divisor of channels that is <= preferred"""
    g = min(preferred, channels)
    while channels % g != 0:
        g -= 1
    return g

# ── Sinusoidal Time Embedding ─────────────────────────────────────────────────
class SinusoidalTimeEmbedding(nn.Module):
    """Converts timestep integer → continuous embedding vector"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = math.log(10000) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t.float()[:, None] * emb[None, :]   # (B, half)
        emb    = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return emb


# ── Self-Attention Block (used at 16×16 resolution) ──────────────────────────
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm  = nn.GroupNorm(8, channels)
        self.qkv   = nn.Conv2d(channels, channels * 3, 1)
        self.proj  = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        B, C, H, W = x.shape
        h   = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]

        attn = torch.bmm(q.permute(0, 2, 1), k) * self.scale   # (B, HW, HW)
        attn = attn.softmax(dim=-1)

        out = torch.bmm(v, attn.permute(0, 2, 1))               # (B, C, HW)
        out = out.reshape(B, C, H, W)
        return x + self.proj(out)


# ── Residual Conv Block with Time Conditioning ────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, use_attn=False):
        super().__init__()
        # FIX: Dynamically determine group sizes to avoid divisibility crashes
        g1 = 8 if in_ch % 8 == 0 else 1
        g2 = 8 if out_ch % 8 == 0 else 1

        self.norm1  = nn.GroupNorm(g1, in_ch)
        self.conv1  = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2  = nn.GroupNorm(g2, out_ch)
        self.conv2  = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_fc = nn.Linear(time_dim, out_ch)
        self.skip    = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act     = nn.SiLU()
        self.attn    = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, t_emb):
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # inject time embedding
        h = h + self.time_fc(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        h = h + self.skip(x)
        return self.attn(h)


# ── U-Net Denoising Backbone ─────────────────────────────────────────────────
class UNet(nn.Module):
    """
    U-Net for DDPM denoising.
    Self-attention is applied at 16×16 resolution as per the spec.
    """
    def __init__(self, img_size=64, in_channels=3, base_ch=64, time_dim=256):
        super().__init__()
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(base_ch),
            nn.Linear(base_ch, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Encoder
        self.enc1 = ResBlock(in_channels, base_ch,     time_dim)                     # 64×64
        self.enc2 = ResBlock(base_ch,     base_ch * 2, time_dim)                     # 32×32
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_dim, use_attn=True)     # 16×16 ← attn
        self.enc4 = ResBlock(base_ch * 4, base_ch * 8, time_dim)                     # 8×8

        self.down = nn.MaxPool2d(2)

        # Bottleneck
        self.mid1 = ResBlock(base_ch * 8, base_ch * 8, time_dim, use_attn=True)
        self.mid2 = ResBlock(base_ch * 8, base_ch * 8, time_dim)

        # Decoder (skip connections from encoder)
        self.up4   = nn.ConvTranspose2d(base_ch * 8, base_ch * 8, 2, 2)
        self.dec4  = ResBlock(base_ch * 16, base_ch * 4, time_dim)

        self.up3   = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 2, 2)
        self.dec3  = ResBlock(base_ch * 8,  base_ch * 2, time_dim, use_attn=True)  # 16×16

        self.up2   = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 2, 2)
        self.dec2  = ResBlock(base_ch * 4,  base_ch,     time_dim)

        self.up1   = nn.ConvTranspose2d(base_ch,     base_ch,     2, 2)
        self.dec1  = ResBlock(base_ch * 2,  base_ch,     time_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)   # (B, time_dim)

        # Encoder
        e1 = self.enc1(x,          t_emb)          # (B, 64,  64, 64)
        e2 = self.enc2(self.down(e1), t_emb)        # (B, 128, 32, 32)
        e3 = self.enc3(self.down(e2), t_emb)        # (B, 256, 16, 16) ← attn
        e4 = self.enc4(self.down(e3), t_emb)        # (B, 512, 8,  8)

        # Bottleneck
        b = self.mid1(self.down(e4), t_emb)
        b = self.mid2(b,             t_emb)

        # Decoder with skip connections
        d = self.dec4(torch.cat([self.up4(b),  e4], dim=1), t_emb)
        d = self.dec3(torch.cat([self.up3(d),  e3], dim=1), t_emb)
        d = self.dec2(torch.cat([self.up2(d),  e2], dim=1), t_emb)
        d = self.dec1(torch.cat([self.up1(d),  e1], dim=1), t_emb)

        return self.out_conv(F.silu(self.out_norm(d)))


# ── DDPM Forward / Reverse Process ───────────────────────────────────────────
class DDPM:
    """
    Manages the noise schedule and sampling.
    T=1000 linear beta schedule as per the original Ho et al. paper.
    """
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.T      = timesteps
        self.device = device

        betas              = torch.linspace(beta_start, beta_end, timesteps).to(device)
        alphas             = 1.0 - betas
        alphas_cumprod     = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register = {
            "betas":             betas,
            "alphas":            alphas,
            "alphas_cumprod":    alphas_cumprod,
            "alphas_cumprod_prev": alphas_cumprod_prev,
            "sqrt_alphas_cumprod":       alphas_cumprod.sqrt(),
            "sqrt_one_minus_alphas_cumprod": (1 - alphas_cumprod).sqrt(),
            "posterior_variance":
                betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod),
        }

    def _get(self, key, t, shape):
        vals = self.register[key][t]
        return vals.reshape(shape[0], *([1] * (len(shape) - 1)))

    def q_sample(self, x0, t, noise=None):
        """Forward process: add noise to x0 at timestep t"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self._get("sqrt_alphas_cumprod", t, x0.shape)
        sqrt_1m    = self._get("sqrt_one_minus_alphas_cumprod", t, x0.shape)
        return sqrt_alpha * x0 + sqrt_1m * noise, noise

    def p_losses(self, model, x0, t):
        """Compute the simple MSE loss for training"""
        noisy_x, noise = self.q_sample(x0, t)
        pred_noise     = model(noisy_x, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, model, x, t_idx):
        """Single reverse step: x_t → x_{t-1}"""
        t   = torch.full((x.size(0),), t_idx, device=self.device, dtype=torch.long)
        eps = model(x, t)

        beta        = self._get("betas",          t, x.shape)
        alpha       = self._get("alphas",         t, x.shape)
        alpha_cum   = self._get("alphas_cumprod", t, x.shape)
        sqrt_1m_ac  = self._get("sqrt_one_minus_alphas_cumprod", t, x.shape)

        mean = (1 / alpha.sqrt()) * (x - beta / sqrt_1m_ac * eps)

        if t_idx > 0:
            var   = self._get("posterior_variance", t, x.shape)
            noise = torch.randn_like(x)
            return mean + var.sqrt() * noise
        return mean

    @torch.no_grad()
    def sample(self, model, n_samples, img_size=64, channels=3):
        """Full reverse diffusion — generate images from noise"""
        model.eval()
        x = torch.randn(n_samples, channels, img_size, img_size, device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t)
        return x.clamp(-1, 1)


def train_ddpm(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    d_cfg    = cfg["ddpm"]
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    img_size = d_cfg["image_size"]
    epochs   = d_cfg["epochs"]
    lr       = d_cfg["lr"]
    bs       = d_cfg["batch_size"]
    T        = d_cfg["timesteps"]

    os.makedirs(cfg["paths"]["save_dir"],    exist_ok=True)
    os.makedirs(cfg["paths"]["samples_dir"], exist_ok=True)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    ds = datasets.ImageFolder(cfg["data"]["train_dir"], transform=tf)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)

    model = UNet(img_size=img_size).to(device)
    ddpm  = DDPM(timesteps=T, device=device)

    total = sum(p.numel() for p in model.parameters())
    print(f"\nDDPM U-Net params: {total:,}")
    print(f"Timesteps: {T} | Image size: {img_size} | Device: {device}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses    = []

    for epoch in range(epochs):
        epoch_loss = 0.0

        for imgs, _ in dl:
            imgs = imgs.to(device)
            t    = torch.randint(0, T, (imgs.size(0),), device=device).long()

            loss = ddpm.p_losses(model, imgs, t)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dl)
        losses.append(avg_loss)
        print(f"Epoch [{epoch+1:>3}/{epochs}]  Loss: {avg_loss:.5f}")

        # generate samples every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            samples = ddpm.sample(model, n_samples=16,
                                  img_size=img_size) * 0.5 + 0.5
            save_image(
                samples,
                os.path.join(cfg["paths"]["samples_dir"],
                             f"ddpm_epoch_{epoch+1}.png"),
                nrow=4,
            )

    torch.save(model.state_dict(),
               os.path.join(cfg["paths"]["save_dir"], "ddpm_unet.pt"))
    print("\nDDPM training complete.")
    return model, ddpm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train_ddpm(args.config)