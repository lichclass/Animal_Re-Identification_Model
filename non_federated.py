import ast
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import pprint

from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score
from collections import Counter

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
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

        return img, self.train_labels[idx], self.eval_labels[idx] , self.encounter_labels[idx]


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
    
    # Values of encounter based are: 'major_vote', 'emb_avg', None

    query_embs = query_embs.to(device)
    query_labels = query_labels.to(device)
    query_encounters = query_encounters.to(device)
    gallery_embs = gallery_embs.to(device)
    gallery_labels = gallery_labels.to(device)
    gallery_encounters = gallery_encounters.to(device)


    # Embedding Averaging per Encounter
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

    # Majority Vote per Encounter
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
    
    # Standard per-Image Evaluation
    else:
        for i in range(num_queries):
            q_lab, q_enc = query_labels[i], query_encounters[i]
            valid_mask = (gallery_encounters != q_enc)

            if not (gallery_labels[valid_mask] == q_lab).any():
                continue

            valid_queries += 1
            query_sims = sim_matrix[i][valid_mask].cpu().numpy()
            query_gt = (gallery_labels[valid_mask] == q_lab).cpu().numpy()

            sorted_indices = np.argsort(query_sims)[::-1]
            sorted_gt = query_gt[sorted_indices]

            if sorted_gt[0]: rank1_count += 1
            if sorted_gt[:5].any(): rank5_count += 1
            ap_sum += average_precision_score(query_gt, query_sims)
    
    if valid_queries == 0:
        return 0.0, 0.0, 0.0
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
    
# ==================== non_federated.py ====================
def run():
    torch.set_float32_matmul_precision('high')

    RUN_TEST_ONLY = True

    args = {
        # Global Config
        'seed': 42,

        # Data Config
        'split_mode': 'closed', # Valid values are 'closed' & 'open'
        'segment': 'head', # Valid values are 'flipper', 'head', 'turtle', 'full'

        # Logging Config
        'dataset_dir': './data/turtle-data',
        'results_path': './results',
        'max_points_eval': 2000,
        
        # Model Config
        'backbone': 'convnext',
        'head': 'adaface',
        'embedding_dim': 512,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',

        # For ArcFace & AdaFace head
        'margin': 0.35,
        'scale': 64.0,

        # For AdaFace head
        't_alpha': 0.01,
        'concentration': 0.2,

        # Train Configs
        'optimizer': 'adamw',
        'criterion': None,
        'scheduler': None,
        'warmup_epochs': 5,
        'num_epochs': 100,
        'early_stopping_patience': 10,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'momentum': 0.9,

        # DataLoader Configs
        'batch_size': 128,
        'num_workers': 8,
        'pin_memory': True,
        'persistent_workers': True,
    }

    print("Data: Configs:")
    print("Segment: ", args['segment'])
    print("Split Mode: ", args['split_mode'])

    # Set seed 
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

    # Global Identity to Label Mapping
    all_identities = sorted(df['identity'].unique().tolist())
    eval_id_to_idx = {identity: idx for idx, identity in enumerate(all_identities)}
    df['eval_label'] = df['identity'].map(eval_id_to_idx)

    # Train Identity to Label Mapping
    train_df_raw = df[df[f'split_{args["split_mode"]}'] == 'train']
    train_identities = sorted(train_df_raw['identity'].unique().tolist())
    train_id_to_idx = {identity: idx for idx, identity in enumerate(train_identities)}
    df['train_label'] = df['identity'].map(train_id_to_idx)

    # Generate datasets
    train_set, val_set, test_set = build_dataset_splits(df, args['split_mode'])
    print("Train/Val/Test sizes:", len(train_set), len(val_set), len(test_set))

    num_train_classes = len(train_id_to_idx)

    # Generate dataloaders
    train_loader = DataLoader(train_set, batch_size=args['batch_size'], shuffle=True, num_workers=args['num_workers'],
                              pin_memory=args['pin_memory'], persistent_workers=args['persistent_workers'])
    val_loader = DataLoader(val_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'],
                            pin_memory=args['pin_memory'], persistent_workers=args['persistent_workers'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], shuffle=False, num_workers=args['num_workers'],
                             pin_memory=args['pin_memory'], persistent_workers=args['persistent_workers'])

    # Build model
    head_kwargs = {}
    if args['head'] == 'arcface':
        head_kwargs = {'s': args['scale'], 'm': args['margin']}
    elif args['head'] == 'adaface':
        head_kwargs = {'m': args['margin'], 'h': args['concentration'], 's': args['scale'], 't_alpha': args['t_alpha']}
    
    model = build_model(
        embedding_dim=args['embedding_dim'],
        num_classes=num_train_classes,
        backbone_type=args['backbone'],
        head_type=args['head'],
        dropout=args['dropout'],
        **head_kwargs
    )
    
    model.to(args['device'])

    if not RUN_TEST_ONLY:
        
        # Setup optimizer and criterion
        if args['optimizer'] == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args['learning_rate'], momentum=args['momentum'], weight_decay=args['weight_decay'])
        elif args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args['learning_rate'], weight_decay=args['weight_decay'])
        
        criterion = nn.CrossEntropyLoss()
        args['criterion'] = 'cross_entropy'

        # Setup scheduler
        warmup_epochs = args['warmup_epochs']
        main_scheduler = CosineAnnealingLR(optimizer, T_max=args['num_epochs'] - warmup_epochs, eta_min=1e-6)
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_epochs])
        args['scheduler'] = 'cosine annealing with linear warmup (SequentialLR)'

        # Training Loop
        best_rank1 = 0.
        best_rank5 = 0.
        best_mAP = 0.
        best_epoch = 0
        patience_counter = 0
        history = {
            'val_rank1': [],
            'val_rank5': [],
            'val_map': [],
        }
        epoch = 0
        for epoch in range(1, args['num_epochs'] + 1):
            model.train()
            running_loss = 0.
            iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{args['num_epochs']} - Training", total=len(train_loader))
            for images, train_labels, _, _ in iterator:
                images = images.to(args['device'], non_blocking=True)
                train_labels = train_labels.to(args['device'], non_blocking=True)
                optimizer.zero_grad()
                with autocast_mode.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(images, train_labels)
                    loss = criterion(logits, train_labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * images.size(0)
                iterator.set_postfix({'Loss': f"{running_loss / ((iterator.n + 1) * args['batch_size']):.4f}"})
            scheduler.step()
            epoch_loss = running_loss / len(train_set)
            print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")

            # Re-Identification Evaluation with Validation Set
            val_embs, val_labels, val_encounters = extract_embeddings(model, val_loader, args['device'], set_name='Validation')
            rank1, rank5, mAP = compute_rank1_rank5_map(
                val_embs, val_labels, val_encounters,
                val_embs, val_labels, val_encounters,
                args['device']
            )
            print(f"Epoch {epoch} Validation Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%, mAP: {mAP*100:.2f}%")
            history["val_rank1"].append(rank1)
            history["val_rank5"].append(rank5)
            history["val_map"].append(mAP)

            # Save best model
            if rank1 > best_rank1:
                best_rank1 = rank1
                best_rank5 = rank5
                best_mAP = mAP
                best_epoch = epoch
                torch.save(model.state_dict(), Path(args['results_path']) / 'best_model.pth')
                print(f"✅ New best model saved at epoch {best_epoch}:")
                print(f"   Rank-1: {best_rank1*100:.2f}%, Rank-5: {best_rank5*100:.2f}%, mAP: {best_mAP*100:.2f}%")
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args['early_stopping_patience']:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break
        print(f"Training completed. Best Validation Rank-1: {best_rank1*100:.2f}%, Rank-5: {best_rank5*100:.2f}%, mAP: {best_mAP*100:.2f}% at epoch {best_epoch}.")
    
    # Final Evaluation on Test Set

    # For Evaluating
    MODEL_PATH = Path("results_data/FINAL4_convnext_adaface_closed_head/best_model.pth")
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    test_embs, test_labels, test_encounters = extract_embeddings(model, test_loader, args['device'], set_name='Test (Using Majority Vote per Encounter Evaluation)')
    rank1, rank5, mAP = compute_rank1_rank5_map(
        test_embs, test_labels, test_encounters,
        test_embs, test_labels, test_encounters,
        args['device'],
        encounter_based='major_vote'
    )
    print(f"Final Test Set Results - Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%, mAP: {mAP*100:.2f}%")
    with open(Path(args['results_path']) / "test_results.txt", "w") as f:
        f.write("(Using Majority Vote per Encounter Evaluation)\n\n")
        f.write(f"Re-identification Test Set Results:\n")
        f.write(f"Rank-1: {rank1*100:.2f}%\n")
        f.write(f"Rank-5: {rank5*100:.2f}%\n")
        f.write(f"mAP: {mAP*100:.2f}%\n")
        # f.write(f"Best Validation Epoch: {best_epoch}\n")
        # f.write(f"Final epoch: {epoch}\n\n")
        f.write(f"Configs:\n")
        pprint.pprint(args, stream=f, indent=4)

    emb, lab = test_embs.cpu().numpy(), test_labels.cpu().numpy()
    max_points = args['max_points_eval']
    if emb.shape[0] > max_points:
        idx = np.random.RandomState(args['seed']).choice(emb.shape[0], max_points, replace=False)
        emb = emb[idx]
        lab = lab[idx]

    plot_tsne(emb, lab, title='t-SNE of Test Set Embeddings', save_path=Path(args['results_path']) / 'tsne_test_set.png')

run()