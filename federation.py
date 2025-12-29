import ast
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import grad_scaler, autocast_mode
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import swin_b, Swin_B_Weights


#==================== dataset.py ====================
class SeaTurtleDataset(Dataset):
    def __init__(self, df, split_mode, split, transform=None):
        subset = df[df[f'split_{split_mode}'] == split].copy()
        
        self.file_names = subset["file_name"].values
        self.train_labels = subset["train_label"].fillna(-1).values.astype(int)
        self.eval_labels = subset["eval_label"].values.astype(int)
        self.encounter_labels = subset["encounter_label"].values.astype(int)
        
        self.bboxes = []
        for b in subset["bounding_box"]:
            if isinstance(b, str) and b.strip():
                try:
                    self.bboxes.append(ast.literal_eval(b))
                except:
                    self.bboxes.append(None)
            elif isinstance(b, (list, tuple)):
                self.bboxes.append(b)
            else:
                self.bboxes.append(None)
                
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        img = Image.open(img_path).convert("RGB")
        bbox = self.bboxes[idx]
        
        if bbox is not None and len(bbox) == 4:
            x, y, w, h = bbox
            W, H = img.size
            
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
        
        if self.transform:
            img = self.transform(img)

        return img, self.train_labels[idx], self.eval_labels[idx] , self.encounter_labels[idx]


#==================== utils.py ====================
@torch.no_grad()
@torch.no_grad()
def extract_embeddings(model, loader, device, set_name=None, max_points=None):
    model.eval()
    all_embs = []
    all_labels = []
    all_encounters = []
    collected = 0
    
    iterator = tqdm(
        loader,
        desc=f'Extracting {set_name}' if set_name else 'Extracting',
    )
    
    for images, _, eval_labels, encounter_ids in iterator:
        images = images.to(device, non_blocking=True)

        _, emb = model(images, None)
        emb = emb.detach().cpu()

        if max_points:
            remaining = max_points - collected
            if remaining <= 0: 
                break
            if emb.size(0) > remaining:
                emb = emb[:remaining]
                eval_labels = eval_labels[:remaining]
                encounter_ids = encounter_ids[:remaining]

        all_embs.append(emb)
        all_labels.append(eval_labels.clone()) 
        all_encounters.append(encounter_ids.clone())

        collected += emb.size(0)
        if max_points and collected >= max_points:
            break

    embs = torch.cat(all_embs, dim=0)
    labels = torch.cat(all_labels, dim=0)
    encounters = torch.cat(all_encounters, dim=0)

    return embs, labels, encounters


@torch.no_grad()
def compute_rank1_rank5_map(query_embs, query_labels, query_encounters, gallery_embs, gallery_labels, gallery_encounters, device):
    query_embs = query_embs.to(device)
    query_labels = query_labels.to(device)
    query_encounters = query_encounters.to(device)
    gallery_embs = gallery_embs.to(device)
    gallery_labels = gallery_labels.to(device)
    gallery_encounters = gallery_encounters.to(device)

    sim_matrix = torch.mm(query_embs, gallery_embs.t())

    num_queries = query_embs.size(0)
    rank1_count = 0
    rank5_count = 0
    ap_sum = 0.0
    valid_queries = 0

    for i in range(num_queries):
        q_lab = query_labels[i]
        q_enc = query_encounters[i]

        valid_mask = (gallery_encounters != q_enc)

        is_positive = (gallery_labels == q_lab)
        if not (is_positive & valid_mask).any():
            continue

        valid_queries += 1

        query_sims = sim_matrix[i][valid_mask].cpu().numpy()
        query_ground_truth = (gallery_labels[valid_mask] == q_lab).cpu().numpy()

        sorted_indices = np.argsort(query_sims)[::-1]
        sorted_gt = query_ground_truth[sorted_indices]

        if sorted_gt[0]:
            rank1_count += 1
        if sorted_gt[:5].any():
            rank5_count += 1

        ap_sum += average_precision_score(query_ground_truth, query_sims)

    if valid_queries == 0:
        return 0.0, 0.0, 0.0

    return rank1_count / valid_queries, rank5_count / valid_queries, ap_sum / valid_queries


def build_dataset_splits(df, split_mode):
    transforms_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = SeaTurtleDataset(df, split_mode, 'train', transform=transforms_train)
    val_set = SeaTurtleDataset(df, split_mode, 'valid', transform=transforms_test)
    test_set = SeaTurtleDataset(df, split_mode, 'test', transform=transforms_test)
    return train_set, val_set, test_set


def build_model(embedding_dim, num_classes, backbone_type='swin', head_type='arcface', pretrained_backbone=True, dropout=0.1, **head_kwargs):
    backbone = build_backbone(embedding_dim=embedding_dim, model_type=backbone_type, pretrained=pretrained_backbone, dropout=dropout)
    head = build_head(embedding_size=embedding_dim, num_classes=num_classes, head_type=head_type, **head_kwargs)

    class ReIDModel(nn.Module):
        def __init__(self, backbone, head, head_type='arcface'):
            super().__init__()
            self.backbone = backbone
            self.head = head
            self.head_type = head_type

        def forward(self, x, labels=None):
            if self.head_type == 'adaface':
                emb, norms = self.backbone(x, return_norms=True)
                logits, emb = self.head(emb, norms, labels)
            elif self.head_type == 'arcface':
                emb = self.backbone(x, return_norms=False)
                logits, emb = self.head(emb, labels)
            return logits, emb

    model = ReIDModel(backbone, head, head_type)
    return model


def plot_tsne(embeddings, labels, title='t-SNE Embeddings', save_path=None):
    tsne = TSNE(n_components=2, perplexity=30, max_iter=3000, random_state=42)
    embs_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(12, 10))
    plt.scatter(embs_2d[:, 0], embs_2d[:, 1], c=labels, cmap='nipy_spectral', s=5, alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    if save_path:
        plt.savefig(save_path, dpi=300)


def plot_loss_curve(history, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def distribute_train_data_to_clients(train_df, split_mode, num_clients):
    clients_data = 


#==================== backbone.py ====================
def build_backbone(embedding_dim=512, model_type='convnext', pretrained=True, dropout=0.1):
    if model_type == 'convnext':
        return ConvNeXtBackbone(embedding_dim, pretrained, dropout)
    if model_type == 'swin':
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


#==================== head.py ====================
def build_head(embedding_size, num_classes, head_type='arcface', **kwargs):
    if head_type == 'arcface':
        return ArcFace(embedding_size, num_classes, **kwargs)
    if head_type == 'adaface':
        return AdaFace(embedding_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


# ArcFace implementation
class ArcFace(nn.Module):
    def __init__(self, embedding_size, num_classes, s=64.0, m=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.s = s
        self.eps = 1e-4

    def forward(self, embeddings, label):
        kernel_norm = F.normalize(self.kernel, dim=0)

        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        if label is None:
            return cosine * self.s, embeddings

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.view(-1, 1), self.m)

        theta = torch.acos(cosine)
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = torch.cos(theta_m)

        return cosine_m * self.s, embeddings


# AdaFace implementation
class AdaFace(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.4, h=0.333, s=64., t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))

        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        # ema prep
        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embeddings, norms, label):
        kernel_norm = F.normalize(self.kernel, dim=0)
        cosine = torch.mm(embeddings,kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps)

        if label is None:
            return cosine * self.s, embeddings

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        # update batchmean batchstd
        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std =  std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        # Angular Margin
        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular.unsqueeze(1)
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        # Additive Margin
        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add.unsqueeze(1)
        cosine = cosine - m_cos

        # scale
        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, embeddings
    

#==================== federated.py ====================
class FederatedClient:
    def __init__(self, train_df, args):
        self.train_df = train_df
        self.args = args

    def get_dataset(self):
        pass

    def train(self):
        pass
    


#==================== run.py ====================
def run():
    torch.set_float32_matmul_precision('high')

    args = {
        # Global Config
        'seed': 42,

        # Data Config
        'split_mode': 'closed', 
        'segment': 'head', # Valid values are 'flipper', 'head', 'turtle', 'full'

        # Logging Config
        'dataset_dir': '/content/turtle-data',
        'results_path': '/content/results',
        'max_points_eval': 2000,
        
        # Model Config
        'backbone': 'convnext',
        'head': 'adaface',
        'embedding_dim': 512,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'margin': 0.4,
        'scale': 64.0,
        't_alpha': 0.01,
        'concentration': 0.333,

        # Train Configs
        'optimizer': 'adamw',
        'criterion': None,
        'scheduler': None,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # DataLoader Configs
        'batch_size': 128,
        'num_workers': 12,
        'pin_memory': True,
        'persistent_workers': True,
    }

    # Setting seed
    set_seed(args['seed'])

    print("Data: Configs:")
    print("Segment: ", args['segment'])
    print("Split Mode: ", args['split_mode'])

    # Data Loading
    data_dir = Path(args['dataset_dir'])
    metadata = f'metadata_splits.csv' if args['segment'] == 'full' else f'metadata_splits_{args["segment"]}.csv'
    df = pd.read_csv(data_dir / metadata)
    df['file_name'] = df['file_name'].apply(lambda x: str(data_dir / x))

    # Required Columns only
    if args['segment'] == 'full':
        df = df[['file_name', 'identity', 'date', f'split_{args["split_mode"]}']]
    else:
        df = df[['file_name', 'identity', 'date', f'split_{args["split_mode"]}', 'bounding_box']]
    
    # Add Encounter Labels
    df['encounter'] = df['identity'].astype(str) + '_' + df['date'].astype(str)
    unique_encounters = df['encounter'].unique().tolist()
    encounter_id_to_idx = {encounter: idx for idx, encounter in enumerate(unique_encounters)}
    idx_to_encounter_id = {idx: encounter for encounter, idx in encounter_id_to_idx.items()}
    df['encounter_label'] = df['encounter'].map(encounter_id_to_idx)

    # Global Identity to Label Mapping
    all_identities = sorted(df['identity'].unique().tolist())
    eval_id_to_idx = {identity: idx for idx, identity in enumerate(all_identities)}
    df['eval_label'] = df['identity'].map(eval_id_to_idx)

    # Train Identity t0 Label Mapping
    train_df_raw = df[df[f'split_{args["split_mode"]}'] == 'train']
    train_identities = sorted(train_df_raw['identity'].unique().tolist())
    train_id_to_idx = {identity: idx for idx, identity in enumerate(train_identities)}
    df['train_label'] = df['identity'].map(train_id_to_idx)

    # Generate datasets
    train_set, val_set, test_set = build_dataset_splits(df, args['split_mode'])

    # Distribute to clients
    clients = 

if __name__ == "__main__":
    run()