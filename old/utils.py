
import random
import os
import json
import ast
import zipfile
import urllib.request
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.metrics import average_precision_score

import torch
import torch.nn as nn
from torchvision import transforms as T
import torch.nn.functional as F

from backbone import build_backbone
from head import build_head
from dataset import SeaTurtleDataset


def prep_dataframe(args):
    # Data Loading
    data_dir = Path(args.dataset_dir)
    metadata = f'metadata_splits.csv' if args.segment == 'full' else f'metadata_splits_{args.segment}.csv'
    df = pd.read_csv(data_dir / metadata)
    df['file_name'] = df['file_name'].apply(lambda x: str(data_dir / x))

    # Required Columns only
    required_cols = ['file_name', 'identity', 'date', f'split_{args.split_mode}']
    if args.segment != 'full' and 'bounding_box' in df.columns:
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

    print(f"Dataframe prepared with {len(df)} images, {len(all_identities)} identities, and {len(unique_encounters)} encounters.")
    return df


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
                if not (gallery_labels[valid_mask] == q_lab).any(): 
                    continue
                
                sims = sim_matrix[idx][valid_mask]
                top1_idx_in_valid = torch.argmax(sims)
                predicted_label = gallery_labels[valid_mask][top1_idx_in_valid].item()
                votes.append(predicted_label)

            if not votes: 
                continue
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
    all_identities = sorted(train_df['identity'].unique().tolist())
    rng = np.random.RandomState(seed)
    rng.shuffle(all_identities)
    
    num_shared_all = int(len(all_identities) * overlap_ratio)
    shared_all_ids = all_identities[:num_shared_all]
    remaining_ids = all_identities[num_shared_all:]
    
    max_clients_limit = max(1, int(num_clients * max_client_ratio))
    if max_clients_limit >= num_clients:
        max_clients_limit = num_clients - 1

    client_id_map = {i: [] for i in range(num_clients)}
    
    for i in range(num_clients):
        client_id_map[i].extend(shared_all_ids)
        
    for identity in remaining_ids:
        n_sites = rng.randint(1, max_clients_limit + 1)
        
        target_clients = rng.choice(range(num_clients), size=n_sites, replace=False)
        
        for client_idx in target_clients:
            client_id_map[client_idx].append(identity)
            
    client_dfs = []
    for i in range(num_clients):
        client_ids = client_id_map[i]
        client_df = train_df[train_df['identity'].isin(client_ids)].copy().reset_index(drop=True)
        
        client_df['is_shared_all'] = client_df['identity'].isin(shared_all_ids)
        
        if verbose:
            excl_count = 0
            limit_count = 0
            for identity in client_ids:
                if identity in shared_all_ids:
                    continue
                appearances = sum(1 for c_list in client_id_map.values() if identity in c_list)
                if appearances == 1:
                    excl_count += 1
                else:
                    limit_count += 1
            
            print(f"Client {i}: {len(client_df)} images | {len(shared_all_ids)} Global, "
                  f"{limit_count} Limited (max {max_clients_limit} sites), {excl_count} Exclusive")
            
        client_dfs.append(client_df)
        
    print(f"Partitioning complete. Max allowed sites per limited ID: {max_clients_limit}\n")
    return client_dfs


def download_dataset():
    data_dir = "data"
    dataset_dir = "turtle-data"
    dataset_link = "https://github.com/lichclass/Animal_Re-Identification_Model/raw/main/downloads/turtle-data.zip"
    zip_path = os.path.join(data_dir, f"{dataset_dir}.zip")
    
    if not os.path.exists(data_dir):
        print(f"Creating Data Directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    else:
        print(f'Data Directory "{data_dir}" already exists. Skipping creation...')
    dataset_path = os.path.join(data_dir, dataset_dir)
    if os.path.exists(dataset_path):
        print(f'Dataset Directory "{dataset_dir}" already exists. Skipping download...')
        return
    print(f"Downloading Dataset: {dataset_dir} from {dataset_link}")

    try:
        urllib.request.urlretrieve(dataset_link, zip_path)
        print("Download complete.")
    except Exception as e:
        print("Download FAILED:", e)
        return
    print("Extracting dataset...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    except Exception as e:
        print("Extraction FAILED:", e)
        return
    finally:
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

    print("Dataset is ready!")


# Function: builds metadata splits for each segment
# By: Elijah
def build_sea_turtle_metadata(annotations, metadata, dataset_path):
    output_metadata_names = ['turtle', 'flipper', 'head']
    exist_count = 0

    # Changes (November 26, 2025)
    # Check if metadata build splits already exists
    for name in output_metadata_names:
        if os.path.exists(os.path.join(dataset_path, f"metadata_splits_{name}.csv")):
            print(f"metadata_splits_{name}.csv already exists. Skipping...")
            exist_count += 1

    if exist_count == len(output_metadata_names):
        print("Metadata splits already exist. Skipping...")
        return

    # Load annotations if path is provided
    if isinstance(annotations, str):
        with open(annotations, 'r') as f:
            annotations = json.load(f)
    
    # Load metadata if path is provided
    if isinstance(metadata, str):
        metadata_df = pd.read_csv(metadata)
    else:
        metadata_df = metadata.copy()
    
    # Extract images, annotations, and categories from COCO format
    images_dict = {img['id']: img for img in annotations['images']}
    categories_dict = {cat['id']: cat['name'] for cat in annotations['categories']}
    
    # Build a mapping from image_id to list of annotations
    image_to_annotations = {}
    for ann in annotations['annotations']:
        image_id = ann['image_id']
        if image_id not in image_to_annotations:
            image_to_annotations[image_id] = []
        image_to_annotations[image_id].append(ann)
    
    # Create a list to store matched records
    matched_records = []
    
    # Iterate through metadata rows
    for idx, row in metadata_df.iterrows():
        file_name = row['file_name']
        
        # Find matching image in annotations by file_name
        matching_image_id = None
        for img_id, img_info in images_dict.items():
            if img_info['file_name'] == file_name:
                matching_image_id = img_id
                break
        
        # If no match found, add row with no bounding box
        if matching_image_id is None:
            row_dict = row.to_dict()
            row_dict['bounding_box'] = None
            row_dict['category'] = None
            matched_records.append(row_dict)
            continue
        
        # Get all annotations for this image
        anns = image_to_annotations.get(matching_image_id, [])
        
        # If no annotations, add row with no bounding box
        if not anns:
            row_dict = row.to_dict()
            row_dict['bounding_box'] = None
            row_dict['category'] = None
            matched_records.append(row_dict)
        else:
            # Add one row per annotation (bounding box)
            for ann in anns:
                row_dict = row.to_dict()
                # Store bounding box as string representation for CSV
                row_dict['bounding_box'] = str(ann['bbox'])
                row_dict['category'] = categories_dict.get(ann['category_id'], 'unknown')
                matched_records.append(row_dict)
    
    # Convert to DataFrame
    metadata_with_bbox = pd.DataFrame(matched_records)
    
    # Split by category
    metadata_turtle = metadata_with_bbox[metadata_with_bbox['category'] == 'turtle'].copy()
    metadata_flipper = metadata_with_bbox[metadata_with_bbox['category'] == 'flipper'].copy()
    metadata_head = metadata_with_bbox[metadata_with_bbox['category'] == 'head'].copy()
    
    # Save to CSV files
    output_path_turtle = os.path.join(dataset_path, 'metadata_splits_turtle.csv')
    output_path_flipper = os.path.join(dataset_path, 'metadata_splits_flipper.csv')
    output_path_head = os.path.join(dataset_path, 'metadata_splits_head.csv')
    
    metadata_turtle.to_csv(output_path_turtle, index=False)
    metadata_flipper.to_csv(output_path_flipper, index=False)
    metadata_head.to_csv(output_path_head, index=False)
    
    print(f"✓ Created metadata_splits_turtle.csv with {len(metadata_turtle)} records")
    print(f"✓ Created metadata_splits_flipper.csv with {len(metadata_flipper)} records")
    print(f"✓ Created metadata_splits_head.csv with {len(metadata_head)} records")
    
    return metadata_turtle, metadata_flipper, metadata_head


# Function: crop images on bounding boxes
# By: Elijah
def crop_turtle(turtle_id, metadata_path, dataset_path="data/turtle-data"):
    # Load metadata
    combined_path = os.path.join(dataset_path, metadata_path)
    print(f"Loading metadata from {combined_path}")
    metadata_df = pd.read_csv(combined_path)
    
    # Filter by turtle_id (identity column)
    turtle_data = metadata_df[metadata_df['identity'] == turtle_id].copy()
    
    if len(turtle_data) == 0:
        print(f"No data found for turtle ID: {turtle_id}")
        return []
    
    cropped_results = []
    
    # Process each row
    for idx, row in turtle_data.iterrows():
        file_name = row['file_name']
        bbox_str = row['bounding_box']
        
        # Skip if no bounding box
        if pd.isna(bbox_str) or bbox_str == 'None':
            print(f"Skipping {file_name}: No bounding box")
            continue
        
        # Parse bounding box string to list
        try:
            bbox = ast.literal_eval(bbox_str)
        except:
            print(f"Error parsing bounding box for {file_name}: {bbox_str}")
            continue
        
        # Load image
        image_path = os.path.join(dataset_path, file_name)
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        try:
            img = Image.open(image_path)
            
            # COCO format: [x, y, width, height]
            x, y, w, h = bbox
            
            # Ensure coordinates are within image bounds
            img_width, img_height = img.size
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), img_width - x)
            h = min(int(h), img_height - y)
            
            # Crop image (PIL uses box format: left, upper, right, lower)
            cropped_img = img.crop((x, y, x + w, y + h))
            
            # Store result
            result = {
                'cropped_image': cropped_img,
                'file_name': file_name,
                'bounding_box': bbox,
                'category': row.get('category', 'unknown'),
                'identity': row['identity']
            }
            cropped_results.append(result)
            
        except Exception as e:
            print(f"Error processing {file_name}: {str(e)}")
            continue
    
    print(f"Successfully cropped {len(cropped_results)} images for {turtle_id}")
    return cropped_results


# Function: display cropped images in a grid
# By: Elijah
def display_cropped_images(cropped_results, max_display=10):
    if not cropped_results:
        print("No images to display")
        return
    
    n_images = min(len(cropped_results), max_display)
    
    # Calculate grid size
    cols = min(4, n_images)
    rows = (n_images + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle single subplot case
    if n_images == 1:
        axes = np.array([axes])
    axes = axes.flatten() if n_images > 1 else axes
    
    for i in range(n_images):
        result = cropped_results[i]
        ax = axes[i] if n_images > 1 else axes[0]
        
        # Display image
        ax.imshow(result['cropped_image'])
        
        # Create title with info
        category = result['category']
        identity = result['identity']
        bbox = result['bounding_box']
        title = f"{identity} - {category}\nBBox: [{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"
        ax.set_title(title, fontsize=10)
        ax.axis('off')
    
    # Hide unused subplots
    for i in range(n_images, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

    
def inspect_annotations():
    DATA_DIR = "data"
    ANNOTATIONS_PATH = Path(DATA_DIR) / "turtle-data" / "annotations.json"
    sample_idx = 0

    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    file_size = os.path.getsize(ANNOTATIONS_PATH)
    keys = list(annotations.keys())
    categories = [{'id': c['id'], 'name': c['name']} for c in annotations['categories']]
    sample_img = annotations["images"][sample_idx]
    sample_annotations = annotations["annotations"][sample_idx]

    print("=====================================================")
    print("SeaTurtleID2022 Dataset Annotations JSON Contents:")
    print(f"> JSON File Size: {file_size / 1024 / 1024} MB")
    print(f"> List of Keys: {keys}")
    print(f"> List of Categories (count = {len(categories)}): {categories}")
    print(f"> Sample Image Data (index: {sample_idx}): {sample_img}")
    print(f"> Sample Annotation Data (index: {sample_idx}):")
    print(sample_annotations)
    print("=====================================================")

def inspect_metadata():
    DATA_DIR = "data"
    ANNOTATIONS_PATH = Path(DATA_DIR) / "turtle-data" / "metadata_splits.csv"
    samples_len = 5

    data = pd.read_csv(ANNOTATIONS_PATH)
    file_size = os.path.getsize(ANNOTATIONS_PATH)
    columns = list(data.columns)
    num_of_data = len(data)

    print("=====================================================")
    print("SeaTurtleID2022 Metadata CSV Contents:")
    print("CSV File Size: {:.2f} MB".format(file_size / 1024 / 1024))
    print(f"Table Columns: {columns}")
    print(f"Number of Data: {num_of_data}")
    print(f"Sample Data (first {samples_len} rows):")
    print(data.head(samples_len))
    print("=====================================================")