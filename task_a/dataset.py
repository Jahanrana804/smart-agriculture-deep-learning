import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch


def get_transforms(intensity="medium", image_size=224):
    """
    Returns (train_transform, val_transform).

    intensity:
        light  — resize + normalize only
        medium — + flip + small rotation
        heavy  — + color jitter + random crop + vertical flip
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std= [0.229, 0.224, 0.225]
    )

    if intensity == "light":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    elif intensity == "medium":
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    else:  # heavy
        train_tf = transforms.Compose([
            transforms.Resize((int(image_size * 1.15), int(image_size * 1.15))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            normalize,
        ])

    # validation / test — no augmentation ever
    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])

    return train_tf, val_tf


def get_dataloaders(train_dir, val_dir,
                    batch_size=32, num_workers=4,
                    aug_intensity="medium", image_size=224):
    """
    Returns (train_loader, val_loader, class_names).
    Expects ImageFolder structure:
        train_dir/class_name/*.jpg
        val_dir/class_name/*.jpg
    """
    train_tf, val_tf = get_transforms(aug_intensity, image_size)

    train_ds = datasets.ImageFolder(train_dir, transform=train_tf)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_tf)

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_ds):,} images | "
          f"Val: {len(val_ds):,} images | "
          f"Classes: {len(train_ds.classes)}")

    return train_dl, val_dl, train_ds.classes


def get_minority_subset(train_dir, n_per_class=100,
                        batch_size=32, num_workers=4, image_size=224):
    """
    Task B helper — returns a dataloader with only n_per_class images
    per class to simulate data scarcity.
    """
    _, val_tf = get_transforms("light", image_size)
    full_ds = datasets.ImageFolder(train_dir, transform=val_tf)

    indices = []
    class_counts = {}
    for idx, (_, label) in enumerate(full_ds.samples):
        if class_counts.get(label, 0) < n_per_class:
            indices.append(idx)
            class_counts[label] = class_counts.get(label, 0) + 1

    subset_ds = Subset(full_ds, indices)
    loader = DataLoader(
        subset_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    print(f"Minority subset: {len(subset_ds):,} images "
          f"({n_per_class} per class × {len(full_ds.classes)} classes)")
    return loader, full_ds.classes
