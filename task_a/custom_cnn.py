import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single conv block: Conv -> BN -> ReLU -> MaxPool"""
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, padding=1, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    """
    5-block CNN built from scratch as required by Task A.
    Input:  (B, 3, 224, 224)
    Output: (B, num_classes)

    Architecture justification:
    - Doubling channels each block extracts increasingly abstract features
    - BatchNorm stabilises training with deeper stacks
    - MaxPool halves spatial dims each block → manageable compute
    - AdaptiveAvgPool before FC allows any input resolution
    - Dropout 0.5 on FC prevents overfitting on small agri datasets
    """
    def __init__(self, num_classes=38):
        super().__init__()

        # 5 convolutional blocks (spec requires >= 5)
        self.block1 = ConvBlock(3,   32)    # 224 -> 112
        self.block2 = ConvBlock(32,  64)    # 112 -> 56
        self.block3 = ConvBlock(64,  128)   # 56  -> 28
        self.block4 = ConvBlock(128, 256)   # 28  -> 14
        self.block5 = ConvBlock(256, 512)   # 14  -> 7

        self.global_pool = nn.AdaptiveAvgPool2d(1)  # 7 -> 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params:     {total:>12,}")
    print(f"  Trainable params: {trainable:>12,}")
    return total, trainable


if __name__ == "__main__":
    model = CustomCNN(num_classes=38)
    count_parameters(model)
    dummy = torch.randn(2, 3, 224, 224)
    out   = model(dummy)
    print(f"Output shape: {out.shape}")   # (2, 38)
