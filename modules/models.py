import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torchvision.models import swin_b, Swin_B_Weights

class SwinB_Backbone(nn.Module):
    def __init__(self, embedding_dim=512):
        super().__init__()
        self.model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        in_features = self.model.head.in_features
        self.model.head = nn.Identity()
        self.feature_layer = nn.Linear(in_features, embedding_dim)
        
    def forward(self, x):
        features = self.model(x)
        embeddings = self.feature_layer(features)
        return embeddings
    

class ArcFaceLoss(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, s=64.0, m=0.5):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, embeddings, labels, return_logits=False):
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(f"ArcFaceLoss: labels out of range. min={labels.min().item()} max={labels.max().item()} num_classes={self.num_classes}")

        embeddings = F.normalize(embeddings, p=2, dim=1)
        weights = F.normalize(self.weight, p=2, dim=1)

        cosine = F.linear(embeddings, weights).clamp(-1+1e-7, 1-1e-7)
        sine = torch.sqrt(1.0 - cosine * cosine)

        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.s

        loss = self.criterion(output, labels)
        if return_logits:
            return loss, output
        return loss