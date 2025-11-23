import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ResNet18Encoder(nn.Module):
    """
    ResNet18 backbone -> AdaptiveAvgPool -> Linear -> L2-normalized embedding
    """
    def __init__(self, embedding_dim: int = 256, use_pretrained: bool = True):
        super().__init__()
        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        # Keep everything up to (but not including) avgpool and fc
        # Children: conv1, bn1, relu, maxpool, layer1..layer4, avgpool, fc
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # -> [B, 512, H/32, W/32]
        self.pool = nn.AdaptiveAvgPool2d(1)                           # -> [B, 512, 1, 1]
        self.fc = nn.Linear(512, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)   # [B, 512, 7, 7] for 224x224 inputs
        x = self.pool(x)       # [B, 512, 1, 1]
        x = x.flatten(1)       # [B, 512]
        x = self.fc(x)         # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1)
        return x

    def display_info():
        print("ResNet18 backbone -> AdaptiveAvgPool -> Linear -> L2-normalized embedding")
