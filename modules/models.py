import torch
import torch.nn as nn

from torchvision.models import swin_b, Swin_B_Weights

class SwinB_Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.model.head = nn.Identity()
        
    def forward(self, x):
        return self.model(x)
    
class ArcFaceHead(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.50):
        super().__init__()
        self.s = s
        self.m = m
        self.fc = nn.Linear(in_features, out_features, bias=False)
        self.cos_m = torch.cos(torch.tensor(m))
        self.sin_m = torch.sin(torch.tensor(m))
        self.th = torch.cos(torch.tensor(3.14159265) - m)
        self.mm = torch.sin(torch.tensor(3.14159265) - m) * m

    def forward(self, x, labels):
        cosine = self.fc(x)
        sine = torch.sqrt(1.0 - torch.clamp(cosine**2, 0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        
        return output