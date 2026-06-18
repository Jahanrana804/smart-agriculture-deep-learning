"""
Task C — Vision Transformer implementations via timm.
Covers: ViT-B/16, DeiT-Small, Swin-Tiny, Hybrid CNN-ViT.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import timm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_dataloaders
from utils.metrics import evaluate_model, count_parameters
from utils.plotting import plot_training_curves, plot_confusion_matrix


# ── Hybrid CNN-ViT ────────────────────────────────────────────────────────────
class HybridCNNViT(nn.Module):
    """
    ResNet-50 backbone (feature extractor) +
    4-layer lightweight transformer encoder.
    Justified: CNN captures local texture, ViT captures global context.
    """
    def __init__(self, num_classes=38, cnn_feat_dim=2048,
                 transformer_dim=512, nhead=8, num_layers=4):
        super().__init__()
        from torchvision import models

        # ResNet-50 without the final FC layer
        resnet      = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.cnn    = nn.Sequential(*list(resnet.children())[:-2])  # (B, 2048, 7, 7)
        self.pool   = nn.AdaptiveAvgPool2d((7, 7))   # fixed spatial grid = 49 tokens

        # project CNN features to transformer dim
        self.proj   = nn.Linear(cnn_feat_dim, transformer_dim)

        # learnable CLS token + positional embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, 49 + 1, transformer_dim) * 0.02
        )

        # 4-layer transformer encoder
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model    = transformer_dim,
            nhead      = nhead,
            dim_feedforward = transformer_dim * 4,
            dropout    = 0.1,
            batch_first= True,
            norm_first = True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(transformer_dim)
        self.head        = nn.Linear(transformer_dim, num_classes)

        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        # CNN features: (B, 2048, 7, 7)
        feats = self.cnn(x)
        feats = self.pool(feats)

        # flatten spatial → tokens: (B, 49, 2048)
        B, C, H, W = feats.shape
        tokens = feats.permute(0, 2, 3, 1).reshape(B, H * W, C)
        tokens = self.proj(tokens)       # (B, 49, transformer_dim)

        # prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)  # (B, 50, transformer_dim)
        tokens = tokens + self.pos_embed

        # transformer
        out = self.transformer(tokens)   # (B, 50, transformer_dim)
        out = self.norm(out[:, 0])       # CLS token output
        return self.head(out)


# ── Model factory ─────────────────────────────────────────────────────────────
VIT_MODELS = {
    "vit_b16":   "vit_base_patch16_224",
    "deit_small":"deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "hybrid":    None,   # custom class above
}


def get_vit_model(name, num_classes=38, pretrained=True):
    """
    name: vit_b16 | deit_small | swin_tiny | hybrid
    """
    name = name.lower()

    if name == "hybrid":
        return HybridCNNViT(num_classes=num_classes)

    timm_name = VIT_MODELS.get(name)
    if timm_name is None:
        raise ValueError(f"Unknown ViT model '{name}'. Choose: {list(VIT_MODELS)}")

    model = timm.create_model(
        timm_name,
        pretrained   = pretrained,
        num_classes  = num_classes,
        drop_path_rate = 0.1,   # stochastic depth as per spec
    )
    return model


def get_vit_model_info(model, name):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n[{name}]  Total: {total:,}  Trainable: {trainable:,}")


# ── Mixup augmentation ────────────────────────────────────────────────────────
def mixup_data(x, y, alpha=0.8, device="cuda"):
    """Mixup augmentation as required in spec for small datasets"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    idx = torch.randperm(batch_size).to(device)
    mixed_x  = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ── Training loop for ViT ─────────────────────────────────────────────────────
def train_vit(model_name="vit_b16", config_path="configs/config.yaml",
              pretrained=True, use_mixup=False):
    import numpy as np   # needed for mixup

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    v_cfg  = cfg["vit"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n{'='*55}")
    print(f"ViT Training: {model_name} | Pretrained: {pretrained}")
    print(f"{'='*55}")

    train_dl, val_dl, class_names = get_dataloaders(
        train_dir    = cfg["data"]["train_dir"],
        val_dir      = cfg["data"]["val_dir"],
        batch_size   = cfg["data"]["batch_size"],
        num_workers  = cfg["data"]["num_workers"],
        aug_intensity= "medium",
        image_size   = cfg["data"]["image_size"],
    )
    num_classes = len(class_names)

    model = get_vit_model(model_name, num_classes, pretrained=pretrained)
    model = model.to(device)
    get_vit_model_info(model, model_name)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = v_cfg["lr"],
        weight_decay = v_cfg["weight_decay"],
    )

    # cosine schedule with linear warmup
    def lr_lambda(epoch):
        warmup = 5
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (v_cfg["epochs"] - warmup)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=v_cfg["label_smoothing"])

    os.makedirs(cfg["paths"]["save_dir"], exist_ok=True)
    os.makedirs(cfg["paths"]["log_dir"],  exist_ok=True)

    best_val_acc = 0.0
    train_losses = []
    val_accs     = []

    for epoch in range(v_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0
        correct    = 0
        total      = 0

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(device), labels.to(device)

            if use_mixup:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=0.8, device=device)
                optimizer.zero_grad()
                out  = model(imgs)
                loss = mixup_criterion(criterion, out, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                out  = model(imgs)
                loss = criterion(out, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            correct    += (out.argmax(1) == labels).sum().item()
            total      += labels.size(0)

        scheduler.step()

        metrics  = evaluate_model(model, val_dl, device, num_classes)
        val_acc  = metrics["accuracy"]
        train_losses.append(epoch_loss / len(train_dl))
        val_accs.append(val_acc)

        print(f"Epoch [{epoch+1:>3}/{v_cfg['epochs']}] "
              f"Loss: {train_losses[-1]:.4f} | "
              f"Train: {correct/total:.4f} | "
              f"Val: {val_acc:.4f} | "
              f"F1: {metrics['f1']:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = os.path.join(cfg["paths"]["save_dir"],
                                f"{model_name}_best.pt")
            torch.save(model.state_dict(), ckpt)
            print(f"  --> Saved best ({val_acc:.4f}) → {ckpt}")

    plot_training_curves(
        train_losses, val_accs,
        save_path=os.path.join(cfg["paths"]["log_dir"],
                               f"{model_name}_curves.png")
    )

    metrics = evaluate_model(model, val_dl, device, num_classes)
    plot_confusion_matrix(
        metrics["confusion_matrix"], class_names,
        save_path=os.path.join(cfg["paths"]["log_dir"],
                               f"{model_name}_cm.png")
    )

    print(f"\nDone. Best val acc: {best_val_acc:.4f}")
    return model, class_names


if __name__ == "__main__":
    import numpy as np
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",     default="vit_b16",
                        choices=list(VIT_MODELS.keys()))
    parser.add_argument("--config",    default="configs/config.yaml")
    parser.add_argument("--no_pretrain", action="store_true")
    parser.add_argument("--mixup",     action="store_true")
    args = parser.parse_args()

    train_vit(
        model_name = args.model,
        config_path= args.config,
        pretrained = not args.no_pretrain,
        use_mixup  = args.mixup,
    )
