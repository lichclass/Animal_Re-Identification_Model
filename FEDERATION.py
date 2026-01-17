import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- PyTorch/Torchvision Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast_mode, grad_scaler

import torchvision.transforms as T
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights,

    # Methods to compare
    swin_b,
    Swin_B_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

# --- Library Imports ---
from wildlife_datasets.datasets import SeaTurtleID2022
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_datasets.splits import ClosedSetSplit


os.environ['KAGGLE_USERNAME'] = "nashadammuoz"
os.environ['KAGGLE_KEY'] = "KGAT_9f227e36a409b0debe5ee7a27090bd72"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# weight aggregation
# Expected Input Example: 
# weights_list = {
#    'client_1': {'weights': model.state_dict(), 'num_samples': 1000},
#    'client_2': {'weights': model.state_dict(), 'num_samples': 1500},
#    ...
# }

# Formula:
# W_{t+1} = sum((n_k / n) * W_k) for k in clients

# Step 1: Compute the total number of samples
total_samples = sum(client['num_samples'] for client in weights_list.values())

# Step 2: Initialize an empty state_dict for the aggregated weights
agg_weights = model.state_dict()

# Step 3: Aggregate the weights
for client in weights_list.values():
    client_weights = client['weights']
    client_samples = client['num_samples']

    weight_factor = client_samples / total
    for key in agg_weights.keys():
        agg_weights[key] += client_weights[key] * weight_factor

# Step 4: Update the global model with the aggregated weights
global_model.load_state_dict(agg_weights)




def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def extract_features(model, dataset, device, batch_size=16):
    model.eval()
    model = model.to(device)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    all_features = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            features = model(imgs)
            all_features.append(features.cpu().numpy())
    return np.vstack(all_features)

def compute_cosine_similarity(query_features, gallery_features):
    query_norm = query_features / (np.linalg.norm(query_features, axis=1, keepdims=True) + 1e-8)
    gallery_norm = gallery_features / (np.linalg.norm(gallery_features, axis=1, keepdims=True) + 1e-8)
    
    similarity_matrix = np.dot(query_norm, gallery_norm.T)
    return similarity_matrix

def evaluate(model, gallery_set, query_set, device, batch_size=16):
    was_training = model.training
    model.eval()

    gallery_features = extract_features(model, gallery_set, device, batch_size)
    query_features = extract_features(model, query_set, device, batch_size)

    similarity_matrix = compute_cosine_similarity(query_features, gallery_features) 

    query_labels = np.array(query_set.labels_string)
    gallery_labels = np.array(gallery_set.labels_string)

    topk_indices = np.argsort(similarity_matrix, axis=1)[:, ::-1][:, :5] 

    rank1_acc = 0
    rank5_acc = 0
    
    for i, q_label in enumerate(query_labels):
        retrieved_labels = gallery_labels[topk_indices[i]]
        
        if retrieved_labels[0] == q_label:
            rank1_acc += 1
            
        if q_label in retrieved_labels:
            rank5_acc += 1

    rank1_acc = (rank1_acc / len(query_labels)) * 100.0
    rank5_acc = (rank5_acc / len(query_labels)) * 100.0

    if was_training:
        model.train()
    
    return rank1_acc, rank5_acc

def clean_path(p):
    if 'turtles-data/data/' in p:
        return p.replace('turtles-data/data/', '')
    return p


def data_loader(config):
    print("--- Loading Data ---")
    SeaTurtleID2022.get_data(root=config['root'])

    # 1. Load Base Data
    if config['body_part'] is None:
        dataset_df = SeaTurtleID2022(root=config['root'], img_load='bbox').df
    else:
        dataset_df = SeaTurtleID2022(root=config['root'], category_name=config['body_part'], img_load='bbox').df
    
    print(f"Original Dataset Size: {len(dataset_df)}")

    # 2. Load Metadata
    try:
        meta_path = Path(config['root']) / 'turtles-data' / 'data' / 'metadata_splits.csv'
        meta_df = pd.read_csv(meta_path)
    except FileNotFoundError:
        raise FileNotFoundError("Metadata splits file not found.")

    # 3. Merge
    dataset_df['join_key'] = dataset_df['path'].apply(clean_path)
    merged_df = pd.merge(
        dataset_df, 
        meta_df[['file_name', 'split_closed', 'split_open']], 
        left_on='join_key', 
        right_on='file_name',
        how='inner' # Changed to inner to avoid NaNs if keys don't match
    )
    print(f"Merged Dataset Size: {len(merged_df)}")

    # 4. Create Base Splits
    split_col = f'split_{config["set"]}'
    train_df = merged_df[merged_df[split_col] == 'train'].reset_index(drop=True)
    valid_df = merged_df[merged_df[split_col] == 'valid'].reset_index(drop=True)
    test_df = merged_df[merged_df[split_col] == 'test'].reset_index(drop=True)

    print(f"Train: {len(train_df)}, Val: {len(valid_df)}, Test: {len(test_df)}")

    # 5. Helper to safely split Query/Gallery
    def safe_split(df, name):
        if len(df) == 0:
            print(f"WARNING: {name} dataframe is empty!")
            return df, df # Return empty
            
        splitter = ClosedSetSplit(ratio_train=0.5, seed=config['seed'])
        # Pass values directly to avoid index confusion
        splits = splitter.split(df)
        
        if len(splits) == 0:
             print(f"WARNING: Splitter returned no splits for {name}")
             return df, df

        gallery_idx, query_idx = splits[0]
        
        # Verify indices are valid
        if gallery_idx.max() >= len(df) or query_idx.max() >= len(df):
             raise IndexError(f"Splitter returned invalid indices for {name}. Max idx: {gallery_idx.max()}, DF len: {len(df)}")

        gal_df = df.iloc[gallery_idx].reset_index(drop=True)
        qry_df = df.iloc[query_idx].reset_index(drop=True)
        return gal_df, qry_df

    # 6. Apply Split
    val_gallery_df, val_query_df = safe_split(valid_df, "Validation")
    test_gallery_df, test_query_df = safe_split(test_df, "Test")

    # 7. Transforms
    t_train = T.Compose([
        T.Resize((config['image_size'], config['image_size'])),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    t_eval = T.Compose([
        T.Resize((config['image_size'], config['image_size'])),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 8. Create Datasets
    train_set = ImageDataset(train_df, root=config['root'], transform=t_train, col_path='path', col_label='identity')
    val_gallery_set = ImageDataset(val_gallery_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    val_query_set = ImageDataset(val_query_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    test_gallery_set = ImageDataset(test_gallery_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')
    test_query_set = ImageDataset(test_query_df, root=config['root'], transform=t_eval, col_path='path', col_label='identity')  

    return train_set, (val_gallery_set, val_query_set), (test_gallery_set, test_query_set)