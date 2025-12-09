# resnet_aspp.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation.deeplabv3 import ASPP

class ResNet50ASPPEncoder(nn.Module):
    """
    ResNet50 backbone + ASPP from DeepLabv3 (Chen et al. 2017).

    Flow:
        Pretrained ResNet50  (up to last conv) -> ASPP -> Global Avg Pool -> Linear -> embedding

    Args:
        embedding_dim: dimension of the final embedding
        use_pretrained: whether to use ImageNet-pretrained ResNet50
    """
    def __init__(self, embedding_dim: int = 256, use_pretrained: bool = True):
        super().__init__()

        # Backbone
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        # Keep everything up to the last conv layer (before global avg pooling)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # [B, 512, H/32, W/32]

        # ASPP that expects 512-channel input (ResNet18 last feature map)
        self.aspp = ASPP(
            in_channels=512,
            atrous_rates=(12, 24, 36),  # typical DeepLabv3 rates
            out_channels=256,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)      # [B, 512, 7, 7] for 224x224 input
        x = self.aspp(x)          # [B, 256, 7, 7]
        x = self.pool(x)          # [B, 256, 1, 1]
        x = x.flatten(1)          # [B, 256]
        x = self.fc(x)            # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1)
        return x

    def display_info(self):
        print("ResNet50 backbone -> ASPP -> AdaptiveAvgPool -> Linear -> L2-normalized embedding")

