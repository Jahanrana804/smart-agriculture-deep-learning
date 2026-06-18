import torch.nn as nn
from torchvision import models


SUPPORTED_MODELS = [
    "resnet50",
    "efficientnet_b3",
    "vgg16",
    "mobilenet_v2",
    "densenet121",
]


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def get_model(name, num_classes=38):
    """
    Returns pretrained model with:
      - All layers frozen
      - Top 2 blocks unfrozen (domain adaptation per spec)
      - Classification head replaced for num_classes
    """
    name = name.lower()

    # ── ResNet-50 ──────────────────────────────────────────────────
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        _freeze_all(model)
        # unfreeze top 2 blocks: layer3 and layer4
        for p in model.layer3.parameters(): p.requires_grad = True
        for p in model.layer4.parameters(): p.requires_grad = True
        model.fc = nn.Linear(2048, num_classes)
        return model

    # ── EfficientNet-B3 ───────────────────────────────────────────
    elif name == "efficientnet_b3":
        model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1
        )
        _freeze_all(model)
        # features has 9 blocks (0-8); unfreeze last 2
        for p in model.features[6].parameters(): p.requires_grad = True
        for p in model.features[7].parameters(): p.requires_grad = True
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    # ── VGG-16 ────────────────────────────────────────────────────
    elif name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        _freeze_all(model)
        # features has indices 0-30; unfreeze last conv block (24-30)
        for p in model.features[24:].parameters(): p.requires_grad = True
        model.classifier[6] = nn.Linear(4096, num_classes)
        return model

    # ── MobileNet-V2 ──────────────────────────────────────────────
    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        _freeze_all(model)
        # features has 19 blocks (0-18); unfreeze last 2
        for p in model.features[17].parameters(): p.requires_grad = True
        for p in model.features[18].parameters(): p.requires_grad = True
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    # ── DenseNet-121 ──────────────────────────────────────────────
    elif name == "densenet121":
        model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        )
        _freeze_all(model)
        # unfreeze denseblock3 and denseblock4
        for p in model.features.denseblock3.parameters(): p.requires_grad = True
        for p in model.features.denseblock4.parameters(): p.requires_grad = True
        model.classifier = nn.Linear(1024, num_classes)
        return model

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            f"Choose from: {SUPPORTED_MODELS}"
        )


def get_model_info(model, model_name):
    """Print quick summary of trainable vs frozen params"""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    print(f"\n[{model_name}]")
    print(f"  Total:     {total:>12,}")
    print(f"  Trainable: {trainable:>12,}  (unfrozen top-2 blocks + new head)")
    print(f"  Frozen:    {frozen:>12,}")
