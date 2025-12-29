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
        self.df = df[df[f'split_{split_mode}'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "file_name"]
        label = int(self.df.loc[idx, "label"])
        img = Image.open(img_path).convert("RGB")
        bbox = self.df.loc[idx, "bounding_box"]
        encounter_id = int(self.df.loc[idx, "encounter_label"])
        if isinstance(bbox, float) and np.isnan(bbox):
            bbox = None
        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            W, H = img.size
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(1, min(x2, W))
            y2 = max(1, min(y2, H))
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)

        return img, label, encounter_id


#==================== utils.py ====================
@torch.no_grad()
def extract_embeddings(model, loader, device, set_name=None, max_points=None):
    all_embs = []
    all_labels = []
    all_encounters = []
    collected = 0
    iterator = tqdm(
        loader,
        desc=f'Extracting embeddings for {set_name}' if set_name else 'Extracting embeddings',
        total=min(len(loader), max_points) if max_points else len(loader)
    )
    for images, labels, encounter_ids in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        encounter_ids = encounter_ids.to(device, non_blocking=True)

        _, emb = model(images, None)
        emb = emb.detach().cpu()
        labels = labels.detach().cpu()
        encounter_ids = encounter_ids.detach().cpu()

        if max_points:
            remaining = max_points - collected
            if remaining <= 0: break
            if emb.size(0) > remaining:
                emb = emb[:remaining]
                labels = labels[:remaining]
                encounter_ids = encounter_ids[:remaining]

        all_embs.append(emb)
        all_labels.append(labels)
        all_encounters.append(encounter_ids)

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


def partition_data_semi_heterogeneous(df, num_clients=4, overlap_ratio=0.3):
    identities = df['identity'].unique().tolist()
    random.shuffle(identities)
    
    num_ids = len(identities)
    ids_per_client = int(num_ids / num_clients)
    
    # Create partitions
    client_dfs = []
    
    for i in range(num_clients):
        # Core unique IDs for this client
        start_idx = i * ids_per_client
        end_idx = min((i + 1) * ids_per_client, num_ids)
        my_ids = identities[start_idx:end_idx]
        
        # Add some overlapping IDs from random other chunks
        num_overlap = int(len(my_ids) * overlap_ratio)
        overlap_pool = [x for x in identities if x not in my_ids]
        if overlap_pool:
            overlap_ids = random.sample(overlap_pool, min(len(overlap_pool), num_overlap))
            my_ids.extend(overlap_ids)
            
        # Filter DataFrame
        client_df = df[df['identity'].isin(my_ids)].reset_index(drop=True)
        client_dfs.append(client_df)
        
    return client_dfs


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


#==================== federated.py ====================
class FederatedClient:
    def __init__(self, client_id, train_df, all_identities, device, args):
        self.client_id = client_id
        self.device = device
        self.args = args
        self.all_identities = all_identities
        
        # 1. Local Data
        self.train_set, _, _ = build_dataset_splits(train_df, args['split_mode'])
        self.train_loader = DataLoader(self.train_set, batch_size=args['batch_size'], 
                                       shuffle=True, num_workers=2, pin_memory=True)
        
        # 2. Local Head (Persists locally!)
        # We use the global number of classes so indices match, but this client only updates its own classes.
        self.head = build_head(embedding_size=512, num_classes=len(all_identities), 
                               head_type=args['loss_head'], **args['head_kwargs']).to(device)
        
        # Optimizer for Head (persists)
        self.opt_head = optim.AdamW(self.head.parameters(), lr=args['lr_head'])

    def train_epoch(self, backbone, global_prototypes):
        backbone = backbone.to(self.device)
        self.head.train()
        backbone.train()
        
        scaler = grad_scaler.GradScaler()
        criterion = nn.CrossEntropyLoss()
        
        # --- PHASE 1: Train Head (Body Frozen) ---
        for param in backbone.parameters(): param.requires_grad = False
        for param in self.head.parameters(): param.requires_grad = True
        
        # Create a temporary optimizer for this phase if needed, or iterate
        # For simplicity, we run a few epochs on Head
        for _ in range(self.args['local_ep_head']):
            for images, labels, _ in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast_mode.autocast(device_type='cuda'):
                    # Forward Body (No Grad)
                    with torch.no_grad():
                        features = backbone(images, return_norms=False) 
                        if self.args['loss_head'] == 'adaface':
                            # AdaFace needs norms, recalculate or modify backbone to return them
                             _, norms = backbone(images, return_norms=True)
                    
                    # Forward Head
                    if self.args['loss_head'] == 'adaface':
                        logits, _ = self.head(features, norms, labels)
                    else:
                        logits, _ = self.head(features, labels)
                        
                    loss = criterion(logits, labels)
                
                self.opt_head.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.opt_head)
                scaler.update()

        # --- PHASE 2: Train Body (Head Frozen) + Proto Loss ---
        for param in backbone.parameters(): param.requires_grad = True
        for param in self.head.parameters(): param.requires_grad = False
        
        opt_body = optim.AdamW(backbone.parameters(), lr=self.args['lr_body']) # New optimizer for body every round
        
        for _ in range(self.args['local_ep_body']):
            for images, labels, _ in self.train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                with autocast_mode.autocast(device_type='cuda'):
                    if self.args['loss_head'] == 'adaface':
                         features, norms = backbone(images, return_norms=True)
                         logits, _ = self.head(features, norms, labels)
                    else:
                        features = backbone(images, return_norms=False)
                        logits, _ = self.head(features, labels)
                    
                    loss_cls = criterion(logits, labels)
                    
                    # --- FEDPROTO REGULARIZATION ---
                    loss_proto = 0.0
                    if global_prototypes is not None:
                        proto_loss_batch = 0.0
                        count = 0
                        for i, lbl in enumerate(labels):
                            lbl_idx = lbl.item()
                            if lbl_idx in global_prototypes:
                                g_proto = global_prototypes[lbl_idx].to(self.device)
                                
                                # CRITICAL: Normalize features before comparison to match inference behavior
                                # Your backbone outputs normalized features if return_norms=False, 
                                # but let's be explicit to avoid magnitude errors.
                                local_feat = F.normalize(features[i], p=2, dim=0) 
                                global_proto_norm = F.normalize(g_proto, p=2, dim=0)
                                
                                proto_loss_batch += F.mse_loss(local_feat, global_proto_norm)
                                count += 1
                        
                        if count > 0:
                            loss_proto = (proto_loss_batch / count) * self.args['lambda_proto']
                    
                    total_loss = loss_cls + loss_proto
                
                opt_body.zero_grad()
                scaler.scale(total_loss).backward()
                scaler.step(opt_body)
                scaler.update()
                
        # --- Calculate New Local Prototypes ---
        new_prototypes = self.compute_prototypes(backbone)
        
        # Return updated backbone weights and new prototypes
        return backbone.state_dict(), new_prototypes

    @torch.no_grad()
    def compute_prototypes(self, backbone):
        backbone.eval()
        sums = {}
        counts = {}
        
        for images, labels, _ in self.train_loader:
            images = images.to(self.device)
            features = backbone(images, return_norms=False)
            
            for i, lbl in enumerate(labels):
                lbl = lbl.item()
                if lbl not in sums:
                    sums[lbl] = features[i].clone().detach().cpu()
                    counts[lbl] = 1
                else:
                    sums[lbl] += features[i].clone().detach().cpu()
                    counts[lbl] += 1
        
        # Return Tuple: (Prototype_Vector, Count)
        prototypes = {k: (v / counts[k], counts[k]) for k, v in sums.items()}
        return prototypes
    

class FederatedServer:
    def __init__(self, global_backbone, device):
        self.global_backbone = global_backbone.to(device)
        self.global_prototypes = {} # Dict mapping label_idx -> feature_vector
        self.device = device

    def aggregate_weights(self, client_weights):
        """Standard FedAvg for Backbone"""
        avg_weights = {}
        for key in client_weights[0].keys():
            avg_weights[key] = sum(cw[key] for cw in client_weights) / len(client_weights)
        self.global_backbone.load_state_dict(avg_weights)

    def aggregate_prototypes(self, client_prototypes_list):
        new_global_protos = {}
        proto_sums = {}
        proto_counts = {}

        for c_data in client_prototypes_list:
            for label, (vec, count) in c_data.items():
                if label not in proto_sums:
                    proto_sums[label] = vec * count # Weighted sum
                    proto_counts[label] = count
                else:
                    proto_sums[label] += vec * count
                    proto_counts[label] += count
        
        for label in proto_sums:
            new_global_protos[label] = proto_sums[label] / proto_counts[label]
        
        self.global_prototypes = new_global_protos


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


#==================== run.py ====================
def run():

    set_seed(42)

    split_mode = 'closed'
    segment = 'head'

    BACKBONE = 'convnext'
    LOSS_HEAD = 'adaface'

    DATASET_DIR = Path('./data/turtle-data')
    METADATA_FILE = DATASET_DIR / f'metadata_splits_{segment}.csv'
    RESULTS_DIR = Path("./results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_NAME_DIR = RESULTS_DIR / f"{BACKBONE}_{LOSS_HEAD}_{split_mode}_{segment}"
    RESULTS_NAME_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = RESULTS_NAME_DIR / "best_model.pth"
    TSNE_PLOTS_DIR = RESULTS_NAME_DIR / "tsne_plots"
    TSNE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_FILE)
    df['file_name'] = df['file_name'].apply(lambda x: DATASET_DIR / x)
    df = df[['file_name', 'identity', 'date', f'split_{split_mode}', 'bounding_box']]
    
    # Global Label Mapping
    identities = df['identity'].unique().tolist()
    identity_to_index = {identity: idx for idx, identity in enumerate(identities)}
    index_to_identity = {idx: identity for identity, idx in identity_to_index.items()}
    df['label'] = df['identity'].map(identity_to_index)

    df['encounter_identity_date'] = df['identity'].astype(str) + "_" + df['date'].astype(str)
    unique_encounters = df['encounter_identity_date'].unique().tolist()
    encounter_id_to_index = {encounter: idx for idx, encounter in enumerate(unique_encounters)}
    index_to_encounter_id = {idx: encounter for encounter, idx in encounter_id_to_index.items()}
    df['encounter_label'] = df['encounter_identity_date'].map(encounter_id_to_index)

    ARGS = {
        'batch_size': 32,
        'split_mode': 'closed',
        'loss_head': 'adaface',
        'head_kwargs': {'m':0.4, 'h':0.333, 's':64.0, 't_alpha':1.0},
        'lr_head': 1e-3,
        'lr_body': 1e-4,
        'local_ep_head': 2, # Epochs for Phase 1
        'local_ep_body': 3, # Epochs for Phase 2
        'lambda_proto': 1.0, # Weight for prototype loss
        'num_rounds': 20,
        'num_clients': 3,
        'overlap_ratio': 0.2
    }

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_df = df[df[f"split_{ARGS['split_mode']}"] == 'train']
    client_dfs = partition_data_semi_heterogeneous(train_df, ARGS['num_clients'], ARGS['overlap_ratio'])

    global_backbone = build_backbone(embedding_dim=512, model_type=BACKBONE, pretrained=True)
    server = FederatedServer(global_backbone, DEVICE)

    clients = []
    for i in range(ARGS['num_clients']):
        client = FederatedClient(i, client_dfs[i], identities, DEVICE, ARGS)
        clients.append(client)
        print(f"Client {i} initialized with {len(client_dfs[i])} images.")

    _, val_set, test_set = build_dataset_splits(df, ARGS['split_mode'])
    val_loader = DataLoader(val_set, batch_size=64, num_workers=2)

    print("Starting Federated Training...")

    for round_idx in range(1, ARGS['num_rounds'] + 1):
        print(f"\n--- Round {round_idx} ---")
        
        client_weights = []
        client_protos_list = []
        
        # Distribute Global Backbone to Clients
        global_state = server.global_backbone.state_dict()
        
        for client in clients:
            # Load global backbone into client
            # (In simulation, we create a copy to avoid reference issues)
            local_backbone = build_backbone(embedding_dim=512, model_type='convnext', pretrained=False)
            local_backbone.load_state_dict(global_state)
            
            # Train Client
            # Pass global prototypes for regularization
            updated_weights, new_protos = client.train_epoch(local_backbone, server.global_prototypes)
            
            client_weights.append(updated_weights)
            client_protos_list.append(new_protos)
        
        # Server Aggregation
        server.aggregate_weights(client_weights)
        server.aggregate_prototypes(client_protos_list)
        
        # --- Evaluation (Global Backbone + Helper Head or Just Embeddings) ---
        # For ReID, we evaluate the Backbone's embeddings directly using distance metrics
        # We don't need a global head for evaluation, just the embeddings
        server.global_backbone.eval()
        val_embs, val_labels, val_encs = extract_embeddings(server.global_backbone, val_loader, DEVICE, set_name='Val')
        
        r1, r5, map_score = compute_rank1_rank5_map(val_embs, val_labels, val_encs, val_embs, val_labels, val_encs, DEVICE)
        print(f"Round {round_idx} Global Eval -> Rank-1: {r1*100:.2f}%, Rank-5: {r5*100:.2f}%, mAP: {map_score*100:.2f}%")
    
    torch.save(server.global_backbone.state_dict(), MODEL_PATH)
    print("\nTraining Complete. Best model saved.")

    server.global_backbone.load_state_dict(torch.load(MODEL_PATH))
    server.global_backbone.eval()
    test_loader = DataLoader(test_set, batch_size=64, num_workers=2)
    test_embs, test_labels, test_encs = extract_embeddings(server.global_backbone, test_loader, DEVICE, set_name='Test')
    r1, r5, map_score = compute_rank1_rank5_map(test_embs, test_labels, test_encs, test_embs, test_labels, test_encs, DEVICE)
    print(f"\nFinal Test Set Evaluation -> Rank-1: {r1*100:.2f}%, Rank-5: {r5*100:.2f}%, mAP: {map_score*100:.2f}%")
    with open(RESULTS_NAME_DIR / "test_results.txt", "w") as f:
        f.write(f"Re-identification Test Set Results:\n")
        f.write(f"Rank-1: {r1*100:.2f}%\n")
        f.write(f"Rank-5: {r5*100:.2f}%\n")
        f.write(f"mAP: {map_score*100:.2f}%\n")

    emb, lab = test_embs.cpu().numpy(), test_labels.cpu().numpy()
    MAX_POINTS = 2000
    if emb.shape[0] > MAX_POINTS:
        idx = np.random.RandomState(42).choice(emb.shape[0], MAX_POINTS, replace=False)
        emb = emb[idx]
        lab = lab[idx]
    plot_tsne(emb, lab, title="T-SNE of Test Embeddings", save_path=TSNE_PLOTS_DIR / "tsne_query_test.png")

run()