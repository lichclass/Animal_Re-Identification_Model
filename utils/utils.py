import os
import urllib.request
import zipfile
import json
import pandas as pd
import numpy as np
from PIL import Image
import ast
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
from torch.utils.data import Subset


def train_one_epoch(model, loader, optim, loss_fn, description="Train"):
    model.train()
    device = next(model.parameters()).device
    running_loss = 0.0
    correct = 0
    total_samples = 0
    iterator = tqdm(loader, desc=description, leave=False)
    for batch in iterator:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optim.step()
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = outputs.argmax(dim=1),
        correct += (preds == labels).sum().item()
    return running_loss / total_samples, correct / total_samples


def evaluate_one_epoch(model, loader, loss_fn, description="Eval"):
    model.eval()
    device = next(model.parameters()).device
    running_loss = 0.0
    running_acc = 0.0
    total_samples = 0
    iterator = tqdm(loader, desc=description, leave=False)
    with torch.no_grad():
        for batch in iterator:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()
            running_acc += acc.item() * batch_size
            total_samples += batch_size
    loss = running_loss / total_samples
    acc = running_acc / total_samples
    return loss, acc


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


def split_dataset_reid(dataset, query_ratio=0.3, min_samples_per_id=2, seed=42):
    np.random.seed(seed)
    
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        valid_indices = dataset.indices
    else:
        base_dataset = dataset
        valid_indices = list(range(len(dataset)))
    
    identity_to_idx = base_dataset.identity_to_idx
    
    identity_to_indices = {}
    
    for idx in valid_indices:
        row = base_dataset.df.iloc[idx]
        identity_str = row['identity']
        label = identity_to_idx[identity_str]
        
        if label not in identity_to_indices:
            identity_to_indices[label] = []
        identity_to_indices[label].append(idx)
    
    query_indices = []
    gallery_indices = []
    skipped = 0
    
    # Split per identity
    for _, indices in identity_to_indices.items():
        if len(indices) < min_samples_per_id:
            skipped += 1
            continue
        
        indices_array = np.array(indices)
        np.random.shuffle(indices_array)
        
        n_query = max(1, int(len(indices) * query_ratio))
        n_query = min(n_query, len(indices) - 1)  # Ensure at least 1 gallery sample
        
        query_indices.extend(indices_array[:n_query].tolist())
        gallery_indices.extend(indices_array[n_query:].tolist())
    
    print(f"\n=== Re-ID Split Statistics ===")
    print(f"Total identities: {len(identity_to_indices)}")
    print(f"Valid identities (>={min_samples_per_id} samples): {len(identity_to_indices) - skipped}")
    print(f"Skipped identities: {skipped}")
    print(f"Query samples: {len(query_indices)}")
    print(f"Gallery samples: {len(gallery_indices)}")
    print(f"Query:Gallery ratio: {len(query_indices)/(len(gallery_indices) or 1):.2f}\n")
    
    query_dataset = Subset(base_dataset, query_indices)
    gallery_dataset = Subset(base_dataset, gallery_indices)
    
    return query_dataset, gallery_dataset 