# conda env: starch_vit
"""ResNet18 classifier for per-granule species classification, pretrained
on ImageNet. Lighter than the earlier whole-photo ViT (StarchViTClassifier
in model.py) -- a better fit for granule-crop scale (small, low-detail
images, ~3,300 total crops across 4 classes) than a data-hungry transformer.
"""

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class GranuleResNetClassifier(nn.Module):
    def __init__(self, num_classes: int, unfreeze_last_n_blocks: int = 0):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1
        self.backbone = resnet18(weights=weights)
        self.image_mean = weights.transforms().mean
        self.image_std = weights.transforms().std

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        # Freeze everything except the new head by default, then unfreeze the
        # last N residual blocks (layer4, then layer3, ...) if requested --
        # same partial-unfreeze pattern as the earlier ViT work, which found
        # unfreezing helped (0.793 -> 0.854 balanced accuracy there).
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.fc.parameters():
            p.requires_grad = True

        unfreezable_blocks = [self.backbone.layer4, self.backbone.layer3, self.backbone.layer2, self.backbone.layer1]
        for block in unfreezable_blocks[:unfreeze_last_n_blocks]:
            for p in block.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.backbone(x)
