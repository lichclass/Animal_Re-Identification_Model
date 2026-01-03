import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    convnext_base, ConvNeXt_Base_Weights,
    swin_b, Swin_B_Weights
)


def build_backbone(embedding_dim=512, model_type='convnext', pretrained=True, dropout=0.1):
    if model_type == 'convnext':
        return ConvNeXtBackbone(embedding_dim, pretrained, dropout)
    elif model_type == 'swin':
        return SwinTransformerBackbone(embedding_dim, pretrained, dropout)
    else:
        raise ValueError(f"Unsupported backbone type: {model_type}")


# ConvNeXt Backbone
class ConvNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True, dropout=0.1):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)

        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x, return_norms=False):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)

        if return_norms:
            norms = torch.norm(emb, p=2, dim=1, keepdim=True)
            emb_normalized = F.normalize(emb, dim=1)
            return emb_normalized, norms.squeeze()
        else:
            emb = F.normalize(emb, dim=1)
            return emb
    
    def forward_raw(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        return emb


# Swin Transformer Backbone
class SwinTransformerBackbone(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True, dropout=0.1):
        super().__init__()
        weights = Swin_B_Weights.IMAGENET1K_V1 if pretrained else None
        model = swin_b(weights=weights)

        in_features = model.head.in_features
        model.head = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_features, embedding_dim)

    def forward(self, x, return_norms=False):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)

        if return_norms:
            norms = torch.norm(emb, p=2, dim=1, keepdim=True)
            emb_normalized = F.normalize(emb, dim=1)
            return emb_normalized, norms.squeeze()
        else:
            emb = F.normalize(emb, dim=1)
            return emb
    
    def forward_raw(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        return emb