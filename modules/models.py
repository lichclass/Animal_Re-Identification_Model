import torch
import torch.nn as nn

from torchvision.models import swin_b, Swin_B_Weights

class SwinB_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)

    def forward(self, x):
        return self.model(x)