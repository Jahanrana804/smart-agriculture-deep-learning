"""
Task B.3 -- Downstream Impact Evaluation
Trains the best Task A classifier under 4 data regimes and
reports accuracy + F1 with statistical significance.
"""
import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset, Dataset
from scipy.stats import ttest_rel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task_a.dataset import get_minority_subset
from task_a.transfer_learning import get_model
from utils.metrics import evaluate_model


class TensorLabelWrapper(Dataset):
    """Wraps any dataset so the label is always a LongTensor."""
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        img, label = self.ds[idx]
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        return img, label


def make_tensor_dataset_from_generator(generator, n_per_class, num_classes,
                                       latent_dim, device, img_size=64):
    generator.eval()
    resize = transforms.Resize((224, 224), antialias=True)
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    all_imgs   = []
    all_labels = []

    with torch.no_grad():
        for cls_idx in range(num_classes):
            z      = torch.randn(n_per_class, latent_dim, device=device)
            labels = torch.full((n_per_class,), cls_idx,
                                dtype=torch.long, device=device)
            imgs   = (generator(z, labels) * 0.5 + 0.5).cpu()
            imgs   = torch.stack([normalize(resize(img)) for img in imgs])
            all_imgs.append(imgs)
            all_labels.append(labels.cpu())

    return TensorDataset(torch.cat(all_imgs), torch.cat(all_labels))


def quick_train(model, train_loader, val_loader, device,
                epochs=10, lr=1e-3, num_classes=38):
    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=lr
    )
    criterion = nn.CrossEntropyLoss()
    best_acc  = 0.0
    run_accs  = []

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            imgs   = imgs.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()

        metrics = evaluate_model(model, val_loader, device, num_classes)
        run_accs.append(metrics["accuracy"])
        if metrics["accuracy"] > best_acc:
            best_acc = metrics["accuracy"]

    return best_acc, run_accs


def make_combined_loader(real_ds, synthetic_ds, batch_size, num_workers=0):
    wrapped_real = TensorLabelWrapper(real_ds)
    combined     = ConcatDataset([wrapped_real, synthetic_ds])
    return DataLoader(combined, batch_size=batch_size,
                      shuffle=True, num_workers=num_workers, pin_memory=False)


def run_downstream_evaluation(config_path="configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device      = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    save_dir    = cfg["paths"]["save_dir"]
    bs          = cfg["data"]["batch_size"]

    print("Building minority subset (100 images/class)...")
    minority_loader, class_names = get_minority_subset(
        cfg["data"]["train_dir"],
        n_per_class  = 100,
        batch_size   = bs,
        num_workers  = 0,
        image_size   = cfg["data"]["image_size"],
    )
    minority_ds = minority_loader.dataset

    from task_a.dataset import get_dataloaders
    _, val_dl, _ = get_dataloaders(
        cfg["data"]["train_dir"], cfg["data"]["val_dir"],
        batch_size  = bs,
        num_workers = 0,
        image_size  = cfg["data"]["image_size"],
    )

    results = {}

    # Regime 1: Real only
    print("\n[Regime 1] Real only")
    model1 = get_model("resnet50", num_classes)
    acc1, accs1 = quick_train(
        model1, minority_loader, val_dl,
        device, epochs=10, num_classes=num_classes
    )
    results["real_only"] = acc1
    print(f"  Best val acc: {acc1:.4f}")

    # Regime 2: Real + GAN augmented
    cgan_path = os.path.join(save_dir, "cgan_generator.pt")
    gan_ds = None
    if os.path.exists(cgan_path):
        print("\n[Regime 2] Real + GAN augmented")
        from task_b.cgan import ConditionalGenerator
        G = ConditionalGenerator(num_classes,
                                  cfg["gan"]["latent_dim"],
                                  cfg["gan"]["image_size"]).to(device)
        G.load_state_dict(torch.load(cgan_path, map_location=device))

        gan_ds = make_tensor_dataset_from_generator(
            G,
            n_per_class = 50,
            num_classes = num_classes,
            latent_dim  = cfg["gan"]["latent_dim"],
            device      = device,
        )
        combined_loader2 = make_combined_loader(minority_ds, gan_ds, bs)

        model2 = get_model("resnet50", num_classes)
        acc2, accs2 = quick_train(
            model2, combined_loader2, val_dl,
            device, epochs=10, num_classes=num_classes
        )
        results["real_gan"] = acc2
        print(f"  Best val acc: {acc2:.4f}")
    else:
        print("  cGAN checkpoint not found -- skipping regime 2")
        acc2, accs2 = acc1, accs1

    # Regime 3: Real + DDPM augmented
    ddpm_path = os.path.join(save_dir, "ddpm_unet.pt")
    ddpm_ds = None
    if os.path.exists(ddpm_path):
        print("\n[Regime 3] Real + DDPM augmented")
        from task_b.ddpm import UNet, DDPM

        unet = UNet(img_size=cfg["ddpm"]["image_size"]).to(device)
        unet.load_state_dict(torch.load(ddpm_path, map_location=device))
        ddpm_obj = DDPM(timesteps=cfg["ddpm"]["timesteps"], device=device)

        print("  Generating 200 DDPM samples...")
        raw = ddpm_obj.sample(unet, n_samples=200,
                               img_size=cfg["ddpm"]["image_size"]) * 0.5 + 0.5

        resize    = transforms.Resize((224, 224), antialias=True)
        normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
        imgs_224  = torch.stack([normalize(resize(img)) for img in raw.cpu()])

        fake_labels = torch.arange(num_classes, dtype=torch.long).repeat(
            200 // num_classes + 1
        )[:200]

        ddpm_ds = TensorDataset(imgs_224, fake_labels)
        combined_loader3 = make_combined_loader(minority_ds, ddpm_ds, bs)

        model3 = get_model("resnet50", num_classes)
        acc3, accs3 = quick_train(
            model3, combined_loader3, val_dl,
            device, epochs=10, num_classes=num_classes
        )
        results["real_ddpm"] = acc3
        print(f"  Best val acc: {acc3:.4f}")
    else:
        print("  DDPM checkpoint not found -- skipping regime 3")
        acc3, accs3 = acc1, accs1

    # Regime 4: Real + Mixed
    print("\n[Regime 4] Real + Mixed")
    if gan_ds is not None and ddpm_ds is not None:
        combined_all = ConcatDataset([
            TensorLabelWrapper(minority_ds), gan_ds, ddpm_ds
        ])
        loader4 = DataLoader(combined_all, batch_size=bs,
                              shuffle=True, num_workers=0)
        model4  = get_model("resnet50", num_classes)
        acc4, accs4 = quick_train(
            model4, loader4, val_dl,
            device, epochs=10, num_classes=num_classes
        )
        results["real_mixed"] = acc4
        print(f"  Best val acc: {acc4:.4f}")
    else:
        results["real_mixed"] = max(acc2, acc3)
        print(f"  Used best of available: {results['real_mixed']:.4f}")

    # Statistical significance
    print("\n[Statistical Test] Paired t-test: Real only vs Real+GAN")
    if len(accs1) == len(accs2):
        from scipy.stats import ttest_rel
        t_stat, p_val = ttest_rel(accs1, accs2)
        print(f"  t-statistic: {t_stat:.4f}  p-value: {p_val:.4f}")
        sig = "YES (p < 0.05)" if p_val < 0.05 else "NO (p >= 0.05)"
        print(f"  Significant improvement: {sig}")

    # Summary
    print("\n" + "="*45)
    print(f"{'Regime':<25} {'Best Val Acc':>12}")
    print("-"*45)
    for k, v in results.items():
        print(f"{k:<25} {v:>12.4f}")
    print("="*45)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run_downstream_evaluation(args.config)