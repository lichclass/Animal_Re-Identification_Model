import ast
import math
import random
import pprint
import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from torch.amp import autocast_mode, grad_scaler
from torch.utils.data import DataLoader, Dataset
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
from torchvision.models import swin_b, Swin_B_Weights


#==================== dataset.py ====================
class SeaTurtleDataset(Dataset):
    def __init__(self, df, split_mode, split, transform=None):
        subset = df[df[f'split_{split_mode}'] == split].copy()
        self.file_names = subset["file_name"].values
        self.train_labels = np.full(len(subset), -1, dtype=int)
        if "train_label" in subset.columns:
            self.train_labels = subset["train_label"].fillna(-1).values.astype(int)
        self.eval_labels = subset["eval_label"].values.astype(int)
        self.encounter_labels = subset["encounter_label"].values.astype(int)
        self.bboxes = [None] * len(self.file_names)
        if "bounding_box" in subset.columns:
            temp_bboxes = []
            for b in subset["bounding_box"]:
                if isinstance(b, str) and b.strip():
                    try:
                        temp_bboxes.append(ast.literal_eval(b))
                    except:
                        temp_bboxes.append(None)
                elif isinstance(b, (list, tuple)):
                    temp_bboxes.append(b)
                else:
                    temp_bboxes.append(None)
            self.bboxes = temp_bboxes
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
        return img, self.train_labels[idx], self.eval_labels[idx], self.encounter_labels[idx]
    

#==================== utils.py ====================
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
def compute_rank1_rank5_map(query_embs, query_labels, query_encounters, gallery_embs, gallery_labels, gallery_encounters, device, encounter_based=None):
    
    query_embs = query_embs.to(device)
    query_labels = query_labels.to(device)
    query_encounters = query_encounters.to(device)
    gallery_embs = gallery_embs.to(device)
    gallery_labels = gallery_labels.to(device)
    gallery_encounters = gallery_encounters.to(device)

    if encounter_based == 'emb_avg':
        unique_q_encs = torch.unique(query_encounters)
        agg_embs, agg_labels, agg_encs = [], [], []
        for enc in unique_q_encs:
            mask = (query_encounters == enc)
            mean_emb = F.normalize(query_embs[mask].mean(dim=0, keepdim=True), p=2, dim=1)
            agg_embs.append(mean_emb)
            agg_labels.append(query_labels[mask][0:1])
            agg_encs.append(enc.view(1))
        query_embs = torch.cat(agg_embs, dim=0)
        query_labels = torch.cat(agg_labels, dim=0)
        query_encounters = torch.cat(agg_encs, dim=0)
    
    sim_matrix = torch.mm(query_embs, gallery_embs.t())
    num_queries = query_embs.size(0)
    rank1_count, rank5_count, ap_sum, valid_queries = 0, 0, 0.0, 0

    if encounter_based == 'major_vote':
        unique_q_encs = torch.unique(query_encounters)
        for enc in unique_q_encs:
            indices = (query_encounters == enc).nonzero(as_tuple=True)[0]
            votes = []
            q_lab = query_labels[indices[0]]

            for idx in indices:
                valid_mask = (gallery_encounters != enc)
                if not (gallery_labels[valid_mask] == q_lab).any(): continue
                
                sims = sim_matrix[idx][valid_mask]
                top1_idx_in_valid = torch.argmax(sims)
                predicted_label = gallery_labels[valid_mask][top1_idx_in_valid].item()
                votes.append(predicted_label)

            if not votes: continue
            valid_queries += 1
            
            winner = Counter(votes).most_common(1)[0][0]
            if winner == q_lab.item():
                rank1_count += 1
                rank5_count += 1
    else:
        for i in range(num_queries):
            q_lab, q_enc = query_labels[i], query_encounters[i]
            valid_mask = (gallery_encounters != q_enc)

            if not (gallery_labels[valid_mask] == q_lab).any(): continue

            valid_queries += 1
            query_sims = sim_matrix[i][valid_mask].cpu().numpy()
            query_gt = (gallery_labels[valid_mask] == q_lab).cpu().numpy()

            sorted_indices = np.argsort(query_sims)[::-1]
            sorted_gt = query_gt[sorted_indices]

            if sorted_gt[0]: rank1_count += 1
            if sorted_gt[:5].any(): rank5_count += 1
            ap_sum += average_precision_score(query_gt, query_sims)
    
    if valid_queries == 0: return 0.0, 0.0, 0.0
    
    print(f"  Valid queries: {valid_queries}/{num_queries} ({100*valid_queries/num_queries:.1f}%)")
    return rank1_count / valid_queries, rank5_count / valid_queries, ap_sum / valid_queries


def build_dataset_splits(df, split_mode):
    transforms_train = T.Compose([
        T.Resize((384, 384)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transforms_test = T.Compose([
        T.Resize((384, 384)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    train_set = SeaTurtleDataset(df, split_mode, 'train', transform=transforms_train)
    val_set = SeaTurtleDataset(df, split_mode, 'valid', transform=transforms_test)
    test_set = SeaTurtleDataset(df, split_mode, 'test', transform=transforms_test)
    return train_set, val_set, test_set


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


def plot_cmc_curve(history, save_path=None):
    plt.figure(figsize=(12, 8))
    plt.plot(history['val_rank1'], label='Rank-1')
    plt.plot(history['val_rank5'], label='Rank-5')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Top-k curves over Rounds')
    plt.legend()
    if save_path:
        plt.savefig(save_path, dpi=300)


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def partition_train_data(train_df, num_clients, overlap_ratio=0.1, max_client_ratio=0.4, seed=42, verbose=True):
    """
    Partitions data such that overlapping identities share NO duplicate images.
    Each client gets a unique subset of photos for any shared identities.
    """
    all_identities = sorted(train_df['identity'].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_identities)
    
    # 1. Identity Assignment (Same logic as before)
    num_shared_all = int(len(all_identities) * overlap_ratio)
    shared_all_ids = all_identities[:num_shared_all]
    remaining_ids = all_identities[num_shared_all:]
    
    max_clients_limit = max(1, min(int(num_clients * max_client_ratio), num_clients - 1))
    
    client_id_map = {i: [] for i in range(num_clients)}
    for i in range(num_clients):
        client_id_map[i].extend(shared_all_ids)
        
    for identity in remaining_ids:
        n_sites = rng.randint(1, max_clients_limit + 1)
        target_clients = rng.choice(range(num_clients), size=n_sites, replace=False)
        for client_idx in target_clients:
            client_id_map[client_idx].append(identity)

    # 2. IMAGE-LEVEL DISTRIBUTION (New Logic)
    client_dfs = []
    
    # Pre-split images for every identity to ensure zero duplication
    # identity -> list of image indices
    id_to_indices = {id: train_df[train_df['identity'] == id].index.tolist() for id in all_identities}
    for id in id_to_indices:
        rng.shuffle(id_to_indices[id]) # Shuffle photos for each turtle
    
    # Track which index in the image list we are at for each identity
    id_cursor = {id: 0 for id in all_identities}

    for i in range(num_clients):
        selected_indices = []
        client_ids = client_id_map[i]
        
        for identity in client_ids:
            # Determine how many clients share this specific identity
            total_shares = sum(1 for c_list in client_id_map.values() if identity in c_list)
            
            # Get all images for this identity
            all_imgs = id_to_indices[identity]
            
            # Calculate this client's 'slice' of photos (e.g., 1/3 of the photos if shared by 3)
            imgs_per_client = max(1, len(all_imgs) // total_shares)
            
            start = id_cursor[identity]
            # If it's the last client for this ID, take all remaining photos to avoid wasting data
            is_last_client_for_id = (sum(1 for j in range(i + 1) if identity in client_id_map[j]) == total_shares)
            
            if is_last_client_for_id:
                end = len(all_imgs)
            else:
                end = min(start + imgs_per_client, len(all_imgs))
            
            selected_indices.extend(all_imgs[start:end])
            id_cursor[identity] = end # Move cursor for the next client that shares this ID
            
        # Build the final DataFrame for this client
        client_df = train_df.loc[selected_indices].copy().reset_index(drop=True)
        client_df['is_shared_all'] = client_df['identity'].isin(shared_all_ids)
        
        if verbose:
            print(f"Client {i}: {len(client_df)} unique images | IDs: {len(client_ids)}")
            
        client_dfs.append(client_df)
        
    print(f"Partitioning complete. Images distributed across sites without duplication.\n")
    return client_dfs


def analyze_data_distribution(client_dfs, all_identities):
    """Analyze how identities are distributed across clients"""
    identity_client_count = {identity: 0 for identity in all_identities}
    
    for client_df in client_dfs:
        for identity in client_df['identity'].unique():
            identity_client_count[identity] += 1
    
    # Count distribution
    distribution = Counter(identity_client_count.values())
    
    print("\n=== Identity Distribution Analysis ===")
    print(f"Total unique identities: {len(all_identities)}")
    for n_clients in sorted(distribution.keys()):
        count = distribution[n_clients]
        pct = 100 * count / len(all_identities)
        print(f"  Found in {n_clients} client(s): {count} identities ({pct:.1f}%)")
    
    # Calculate heterogeneity metric
    avg_clients_per_id = np.mean(list(identity_client_count.values()))
    print(f"\nAverage clients per identity: {avg_clients_per_id:.2f}")
    print("=" * 40 + "\n")
    
    return identity_client_count


#==================== backbone.py ====================
def build_backbone(embedding_dim=512, model_type='convnext', pretrained=True, dropout=0.1):
    if model_type == 'convnext':
        return ConvNeXtBackbone(embedding_dim, pretrained, dropout)
    elif model_type == 'swin':
        return SwinTransformerBackbone(embedding_dim, pretrained, dropout)
    else:
        raise ValueError(f"Unsupported backbone type: {model_type}")


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


#==================== head.py ====================
def build_head(embedding_size, num_classes, head_type='arcface', **kwargs):
    if head_type == 'arcface':
        return ArcFace(embedding_size, num_classes, **kwargs)
    elif head_type == 'adaface':
        return AdaFace(embedding_size, num_classes, **kwargs)
    else:
        raise ValueError(f"Unsupported head type: {head_type}")


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


class AdaFace(nn.Module):
    def __init__(self, embedding_size, num_classes, m=0.4, h=0.333, s=64., t_alpha=1.0):
        super(AdaFace, self).__init__()
        self.num_classes = num_classes
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, num_classes))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.eps = 1e-3
        self.h = h
        self.s = s

        self.t_alpha = t_alpha
        self.register_buffer('t', torch.zeros(1))
        self.register_buffer('batch_mean', torch.ones(1)*(20))
        self.register_buffer('batch_std', torch.ones(1)*100)

    def forward(self, embeddings, norms, label):
        kernel_norm = F.normalize(self.kernel, dim=0)
        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1+self.eps, 1-self.eps)

        if label is None:
            return cosine * self.s, embeddings

        safe_norms = torch.clip(norms, min=0.001, max=100)
        safe_norms = safe_norms.clone().detach()

        with torch.no_grad():
            mean = safe_norms.mean().detach()
            std = safe_norms.std().detach()
            self.batch_mean = mean * self.t_alpha + (1 - self.t_alpha) * self.batch_mean
            self.batch_std = std * self.t_alpha + (1 - self.t_alpha) * self.batch_std

        margin_scaler = (safe_norms - self.batch_mean) / (self.batch_std+self.eps)
        margin_scaler = margin_scaler * self.h
        margin_scaler = torch.clip(margin_scaler, -1, 1)

        m_arc = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_arc.scatter_(1, label.reshape(-1, 1), 1.0)
        g_angular = self.m * margin_scaler * -1
        m_arc = m_arc * g_angular.unsqueeze(1)
        theta = cosine.acos()
        theta_m = torch.clip(theta + m_arc, min=self.eps, max=math.pi-self.eps)
        cosine = theta_m.cos()

        m_cos = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_cos.scatter_(1, label.reshape(-1, 1), 1.0)
        g_add = self.m + (self.m * margin_scaler)
        m_cos = m_cos * g_add.unsqueeze(1)
        cosine = cosine - m_cos

        scaled_cosine_m = cosine * self.s
        return scaled_cosine_m, embeddings


def build_model(embedding_dim, num_classes, backbone_type='convnext', head_type='arcface', pretrained_backbone=True, dropout=0.1, **head_kwargs):
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


#==================== federated.py ====================
class FederatedClient:
    def __init__(self, client_id, train_df, args):
        self.client_id = client_id
        self.args = args
        self.device = args['device']
        self.train_df = train_df.copy()

        unique_local_ids = sorted(self.train_df['identity'].unique().tolist())
        self.num_local_classes = len(unique_local_ids)
        self.local_id_map = {identity: idx for idx, identity in enumerate(unique_local_ids)}
        self.inv_local_id_map = {idx: identity for identity, idx in self.local_id_map.items()}
        self.train_df['train_label'] = self.train_df['identity'].map(self.local_id_map)

        head_kwargs = {}
        if args['head'] == 'arcface':
            head_kwargs = {'s': args['s'], 'm': args['m']}
        elif args['head'] == 'adaface':
            head_kwargs = {'m': args['m'], 'h': args['h'], 's': args['s'], 't_alpha': args['t_alpha']}
        
        self.model = build_model(
            embedding_dim=args['embedding_dim'],
            num_classes=self.num_local_classes,
            backbone_type=args['backbone'],
            head_type=args['head'],
            pretrained_backbone=True,
            dropout=args['dropout'],
            **head_kwargs
        ).to(self.device)

        print(f"[Client {self.client_id}] Images: {len(self.train_df)} | Identities: {self.num_local_classes}")

    def get_loader(self):
        transform = T.Compose([
            T.Resize((384, 384)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        dataset = SeaTurtleDataset(self.train_df, self.args['split_mode'], 'train', transform)
        return DataLoader(
            dataset, 
            batch_size=self.args['batch_size'], 
            shuffle=True, 
            num_workers=self.args['num_workers'], 
            pin_memory=self.args['pin_memory'],
            persistent_workers=self.args['persistent_workers']
        )

    def train(self, global_backbone_state, global_prototypes, current_lr):
        self.model.backbone.load_state_dict(global_backbone_state)
        self.model.train()
        
        criterion = nn.CrossEntropyLoss()
        loader = self.get_loader()
        scaler = grad_scaler.GradScaler()

        if global_prototypes:
            gpu_prototypes = {
                k: v.to(self.device) for k, v in global_prototypes.items()
            }
        else:
            gpu_prototypes = {}
        
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=current_lr, 
            weight_decay=self.args['weight_decay']
        )
        
        for epoch in range(self.args['local_epochs']):
            epoch_loss = 0.0
            epoch_cls_loss = 0.0
            epoch_proto_loss = 0.0
            
            iterator = tqdm(
                loader,
                desc=f'Client {self.client_id} | Epoch {epoch+1}/{self.args["local_epochs"]}'
            )
            
            for images, train_labels, _, _ in iterator:
                images, train_labels = images.to(self.device), train_labels.to(self.device)
                
                with autocast_mode.autocast(device_type='cuda'):
                    logits, emb = self.model(images, train_labels)
                    loss_cls = criterion(logits, train_labels)
                    
                    # Federated Prototype Loss
                    loss_proto = torch.tensor(0.0, device=self.device)
                    if gpu_prototypes:
                        valid_terms = []
                        for i, local_idx in enumerate(train_labels):
                            # Map local index to global identity string
                            identity_str = self.inv_local_id_map[local_idx.item()]
                            
                            if identity_str in gpu_prototypes:
                                g_proto = gpu_prototypes[identity_str]
                                # MSE between normalized local embedding and global prototype
                                valid_terms.append(F.mse_loss(emb[i], g_proto))
                        
                        if valid_terms:
                            loss_proto = torch.stack(valid_terms).mean() * self.args['lambda_proto']
                    
                    # Combine Losses
                    total_loss = loss_cls + loss_proto
                
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                epoch_loss += total_loss.item()
                epoch_cls_loss += loss_cls.item()
                epoch_proto_loss += loss_proto.item()
                
                iterator.set_postfix({
                    'Loss': f'{total_loss.item():.4f}',
                    'Cls': f'{loss_cls.item():.4f}',
                    'Proto': f'{loss_proto.item():.4f}'
                })
            
            avg_loss = epoch_loss / len(loader)
            avg_cls = epoch_cls_loss / len(loader)
            avg_proto = epoch_proto_loss / len(loader)
            print(f'  [Client {self.client_id}] Epoch {epoch+1}: Loss={avg_loss:.4f} (Cls={avg_cls:.4f}, Proto={avg_proto:.4f})')
        
        # Clear GPU prototypes to free memory
        del gpu_prototypes
        torch.cuda.empty_cache()

        new_prototypes = self._compute_local_prototypes(loader)
        
        # Return backbone weights only
        return self.model.backbone.state_dict(), new_prototypes

    @torch.no_grad()
    def _compute_local_prototypes(self, loader):
        self.model.eval()
        prototype_sums = {}
        prototype_counts = {}
        
        for images, train_labels, _, _ in loader:
            images = images.to(self.device)
            
            # Get Raw Embeddings, no normalization
            raw_emb = self.model.backbone.forward_raw(images)

            train_labels = train_labels.cpu()
            
            for i in range(raw_emb.size(0)):
                local_idx = train_labels[i].item()
                identity_str = self.inv_local_id_map[local_idx]

                emb_i = raw_emb[i].detach().cpu()
                
                if identity_str not in prototype_sums:
                    prototype_sums[identity_str] = emb_i.clone()
                    prototype_counts[identity_str] = 1
                else:
                    prototype_sums[identity_str] += emb_i
                    prototype_counts[identity_str] += 1
        
        prototypes_with_counts = {}
        for identity_str, vec_sum in prototype_sums.items():
            cnt = prototype_counts[identity_str]
            mean_vec = vec_sum / cnt
            proto = F.normalize(mean_vec, p=2, dim=0)
            prototypes_with_counts[identity_str] = (proto, cnt)
        
        return prototypes_with_counts   
    

class FederatedServer:
    def __init__(self, args):
        self.args = args
        self.device = args['device']
        
        self.global_backbone = build_backbone(
            embedding_dim=args['embedding_dim'], 
            model_type=args['backbone'], 
            pretrained=True,
            dropout=args.get('dropout', 0.1)
        ).to(self.device)

        self.global_prototypes = {}

    def _get_eval_model(self):
        backbone = self.global_backbone
        
        class EvalWrapper(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
            
            def forward(self, x, labels=None):
                emb = self.backbone(x, return_norms=False)
                return None, emb  # (logits, embeddings) tuple
        
        return EvalWrapper(backbone)

    def aggregate_weights(self, client_weights_list):
        print("[Server] Aggregating Backbone Weights...")
        avg_weights = {}
        num_clients = len(client_weights_list)
        
        ref_keys = client_weights_list[0].keys()
        for key in ref_keys:
            avg_weights[key] = sum(cw[key] for cw in client_weights_list) / num_clients
        
        self.global_backbone.load_state_dict(avg_weights)
        return avg_weights

    def aggregate_prototypes(self, client_protos_list):
        print("[Server] Aggregating Prototypes (weighted by sample counts)...")

        round_sums = {}
        round_counts = {}

        for c_protos in client_protos_list:
            for identity_str, (proto_vec, n) in c_protos.items():

                n = int(n)
                if n <= 0:
                    continue

                vec = proto_vec.float()

                if identity_str not in round_sums:
                    round_sums[identity_str] = vec * n
                    round_counts[identity_str] = n
                else:
                    round_sums[identity_str] += vec * n
                    round_counts[identity_str] += n

        momentum = self.args["proto_momentum"]
        updated_count = 0
        new_count = 0

        # Update global prototypes with momentum
        for identity_str, vec_sum in round_sums.items():

            total_n = round_counts[identity_str]
            if total_n <= 0:
                continue
            
            # Weighted mean of prototypes from this round
            current_avg = vec_sum / total_n
            current_avg = F.normalize(current_avg, p=2, dim=0)

            if identity_str in self.global_prototypes:
                old_proto = self.global_prototypes[identity_str].float()
                new_proto = (old_proto * momentum) + (current_avg * (1 - momentum))
                self.global_prototypes[identity_str] = F.normalize(new_proto, p=2, dim=0)
                updated_count += 1
            else:
                self.global_prototypes[identity_str] = current_avg
                new_count += 1

        print(f"[Server] Updated: {updated_count} | New: {new_count} | Total: {len(self.global_prototypes)}")
        return self.global_prototypes


    def evaluate(self, loader, set_name="Val"):
        print(f"[Server] Evaluating on {set_name} Set...")
        eval_model = self._get_eval_model()
        embs, labels, encounters = extract_embeddings(
            eval_model, loader, self.device, set_name
        )
        
        r1, r5, mAP = compute_rank1_rank5_map(
            embs, labels, encounters, 
            embs, labels, encounters, 
            self.device, encounter_based=None
        )
        return r1, r5, mAP
    

#==================== main.py ====================
def run():
    torch.set_float32_matmul_precision('high')
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    torch.backends.cudnn.benchmark = True

    RUN_TEST_ONLY = False

    args = {
        # Global Config
        'seed': 42,

        # Data Config
        'split_mode': 'closed', # Valid values are 'closed' & 'open'
        'segment': 'head', # Valid values are 'flipper', 'head', 'turtle', 'full'

        # Logging Config
        'dataset_dir': './data/turtle-data',
        'results_path': './results_data/FINAL18_fedproto_lambda_2_0',
        'max_points_eval': 2000,
        
        # Model Config
        'backbone': 'convnext',
        'head': 'adaface',
        'embedding_dim': 512,
        'dropout': 0.2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # For ArcFace / AdaFace Head
        'm': 0.35,
        's': 64.0,

        # AdaFace specific
        't_alpha': 0.01,
        'h': 0.2,
        
        # Training Configs
        'learning_rate': 1e-4,
        'early_stopping_patience': 5,
        'weight_decay': 1e-4,
        'warmup_rounds': 3,

        # DataLoader Configs
        'batch_size': 128,
        'num_workers': 12,
        'pin_memory': True,
        'persistent_workers': True,
        
        # Federation Configs
        'num_clients': 5,
        'federation_rounds': 50,
        'local_epochs': 5,
        'lambda_proto': 1.0,
        'proto_momentum': 0.0,
        'overlap_ratio': 0.1,
        'eval_every': 1, 
        'max_client_ratio': 0.4,
    }

    print("Data: Configs")
    print("Segment: ", args['segment'])
    print("Split Mode: ", args['split_mode'])
    print("Federation: Clients =", args['num_clients'], ", Rounds =", args['federation_rounds'])

    set_seed(args['seed'])

    # Data Loading
    data_dir = Path(args['dataset_dir'])
    metadata = f'metadata_splits.csv' if args['segment'] == 'full' else f'metadata_splits_{args["segment"]}.csv'
    df = pd.read_csv(data_dir / metadata)
    df['file_name'] = df['file_name'].apply(lambda x: str(data_dir / x))

    # Required Columns only
    required_cols = ['file_name', 'identity', 'date', f'split_{args["split_mode"]}']
    if args['segment'] != 'full' and 'bounding_box' in df.columns:
        required_cols.append('bounding_box')
    df = df[required_cols]

    # Add Encounter Labels
    df['encounter'] = df['identity'].astype(str) + '_' + df['date'].astype(str)
    unique_encounters = df['encounter'].unique().tolist()
    encounter_id_to_idx = {encounter: idx for idx, encounter in enumerate(unique_encounters)}
    idx_to_encounter_id = {idx: encounter for encounter, idx in encounter_id_to_idx.items()}
    df['encounter_label'] = df['encounter'].map(encounter_id_to_idx)

    # Global Identity Mapping (For Evaluation)
    all_identities = sorted(df['identity'].unique().tolist())
    eval_id_to_idx = {identity: idx for idx, identity in enumerate(all_identities)}
    df['eval_label'] = df['identity'].map(eval_id_to_idx)

    # Data Partitioning between clients
    train_df_full = df[df[f'split_{args["split_mode"]}'] == 'train'].reset_index(drop=True)
    client_dfs = partition_train_data(
        train_df=train_df_full, 
        num_clients=args['num_clients'], 
        overlap_ratio=args['overlap_ratio'], 
        max_client_ratio=args['max_client_ratio'], 
        seed=args['seed']
    )
    analyze_data_distribution(client_dfs, all_identities)

    # Server setup
    server = FederatedServer(args)

    # Set and Loaders for Evaluation
    _, val_set, test_set = build_dataset_splits(df, args['split_mode'])
    val_loader = DataLoader(
        val_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=args['pin_memory'],
        persistent_workers=args['persistent_workers']
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args['batch_size'],
        shuffle=False,
        num_workers=args['num_workers'],
        pin_memory=args['pin_memory'],
        persistent_workers=args['persistent_workers']
    )

    if not RUN_TEST_ONLY:
        
        # For Fault Tolerance: Load existing checkpoint if available
        results_path = Path(args['results_path'])
        checkpoint = {}
        if results_path.exists():
            print(f"Results directory {results_path} already exists. Existing files may be overwritten. Taking existing checkpoint")
            MODEL_PATH = results_path / 'checkpoint_backbone.pth'
            if MODEL_PATH.exists():
                print(f"Loading existing model from {MODEL_PATH} to resume training...")
                checkpoint = torch.load(MODEL_PATH)

                server.global_backbone.load_state_dict(checkpoint['backbone'])
                server.global_prototypes = checkpoint['prototypes']
        else:
            results_path.mkdir(parents=True, exist_ok=True)

        clients = [FederatedClient(client_id, client_dfs[client_id], args) for client_id in range(args['num_clients'])]

        dummy_optimizer = optim.AdamW(server.global_backbone.parameters(), lr=args['learning_rate'])
        warmup_rounds = args['warmup_rounds']
        main_scheduler = CosineAnnealingLR(dummy_optimizer, T_max=args['federation_rounds'] - warmup_rounds, eta_min=1e-6)
        warmup_scheduler = LinearLR(dummy_optimizer, start_factor=0.1, total_iters=warmup_rounds)
        scheduler = SequentialLR(dummy_optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_rounds])

        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])

        history = checkpoint.get("history", {"val_rank1": [], "val_rank5": [], "val_mAP": []})
        best_rank1 = checkpoint.get("best_rank1", 0.0)
        best_rank5 = checkpoint.get("best_rank5", 0.0)
        best_mAP = checkpoint.get("best_mAP", 0.0)
        best_round = checkpoint.get("best_round", 0)
        patience_counter = checkpoint.get("patience_counter", 0)
        last_idx = checkpoint.get('round_idx', 0)

        global_weights = server.global_backbone.state_dict()
        global_prototypes = server.global_prototypes

        round_idx = last_idx
        for round_idx in range(last_idx + 1, args['federation_rounds'] + 1):
            print(f"\n=== Federation Round {round_idx}/{args['federation_rounds']} ===")
            current_lr = scheduler.get_last_lr()[0]

            client_weights_list = []
            client_protos_list = []

            for client in clients:
                print(f"Training Client {client.client_id}...")
                c_weights, c_prototypes = client.train(global_weights, global_prototypes, current_lr)

                client_weights_list.append(c_weights)
                client_protos_list.append(c_prototypes)
            scheduler.step()
        
            print("Aggregating updates on server...")
            global_weights = server.aggregate_weights(client_weights_list)
            global_prototypes = server.aggregate_prototypes(client_protos_list)

            # Adding this because i can't use A100 Anymore and have to switch to a smaller GPU
            del client_weights_list
            del client_protos_list
            torch.cuda.empty_cache()

            # Save checkpoint
            torch.save({
                "round_idx": round_idx,
                "backbone": server.global_backbone.state_dict(),
                "prototypes": server.global_prototypes,

                "history": history,
                "best_rank1": best_rank1,
                "best_rank5": best_rank5,
                "best_mAP": best_mAP,
                "best_round": best_round,
                "patience_counter": patience_counter,

                "scheduler": scheduler.state_dict(),
                "args": args,
            }, results_path / "checkpoint_backbone.pth")

            if round_idx % args['eval_every'] == 0:
                val_r1, val_r5, val_mAP = server.evaluate(val_loader, set_name="Validation")
                history['val_rank1'].append(val_r1)
                history['val_rank5'].append(val_r5)
                history['val_mAP'].append(val_mAP)

                if val_r1 > best_rank1:
                    best_rank1 = val_r1
                    best_rank5 = val_r5
                    best_mAP = val_mAP
                    best_round = round_idx
                    torch.save({
                        'backbone': server.global_backbone.state_dict(), 
                        'prototypes': server.global_prototypes,
                    }, Path(args['results_path']) / 'best_backbone.pth'
                    )
                    print(f"✅ New best model saved at round {best_round}:")
                    print(f"   Rank-1: {best_rank1*100:.2f}%, Rank-5: {best_rank5*100:.2f}%, mAP: {best_mAP*100:.2f}%")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= args['early_stopping_patience']:
                        print(f"Early stopping triggered after {round_idx} rounds.")
                        break
                print(f"Rank-1: {val_r1*100:.2f}%, Rank-5: {val_r5*100:.2f}%, mAP: {val_mAP*100:.2f}% at round {round_idx}")

            with open(Path(args['results_path']) / 'training_history.json', 'w') as f:
                json.dump(history, f, indent=4)

        print(f"Training completed. Best Validation Rank-1: {best_rank1*100:.2f}%, Rank-5: {best_rank5*100:.2f}%, mAP: {best_mAP*100:.2f}% at round {best_round}.")

    # Evaluation on Test Set
    MODEL_PATH = Path(args['results_path']) / 'best_backbone.pth'
    print(f"\nLoading best model from {MODEL_PATH} for final evaluation...")
    checkpoint = torch.load(MODEL_PATH)
    server.global_backbone.load_state_dict(checkpoint['backbone'])
    server.global_prototypes = checkpoint['prototypes']

    rank1, rank5, mAP = server.evaluate(test_loader, set_name="Test")

    print(f"Final Test Set Performance:")
    print(f"  Rank-1: {rank1*100:.2f}%")
    print(f"  Rank-5: {rank5*100:.2f}%")
    print(f"  mAP: {mAP*100:.2f}%")
    with open(Path(args['results_path']) / 'test_results.txt', 'w') as f:
        f.write(f"Test Set Performance:\n")
        f.write(f"Federated Re-Identification Test Set Results:\n")
        f.write(f"Rank-1: {rank1*100:.2f}%\n")
        f.write(f"Rank-5: {rank5*100:.2f}%\n")
        f.write(f"mAP: {mAP*100:.2f}%\n")
        f.write(f"(Best Validation at Round {best_round})\n")
        f.write(f"Final round: {round_idx}\n")
        f.write("Configs: \n")
        pprint.pprint(args, stream=f, indent=4)

    eval_model = server._get_eval_model()
    eval_model.to(args['device'])
    emb, lab, _ = extract_embeddings(eval_model, test_loader, args['device'], set_name="Test")
    emb, lab = emb.cpu().numpy(), lab.cpu().numpy()
    max_points = args['max_points_eval']
    if emb.shape[0] > max_points:
        emb = emb[:max_points]
        lab = lab[:max_points]
    plot_tsne(emb, lab, title='t-SNE Visualization of Test Embeddings', save_path=Path(args['results_path']) / 'tsne_test_set.png')

if __name__ == '__main__':
    run()