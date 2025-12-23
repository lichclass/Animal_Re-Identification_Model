import os
import urllib.request
import zipfile
import json
import pandas as pd
import numpy as np
from PIL import Image
import ast
import matplotlib.pyplot as plt

import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier

import torch
from tqdm import tqdm
from torch.utils.data import Subset


def train_one_epoch_with_accumulation(model, loss_fn, optimizer, train_loader, 
                                      device, epoch, total_epochs, accumulation_steps):
    """
    Train for one epoch with gradient accumulation
    """
    model.train()
    loss_fn.train()
    
    total_loss = 0
    correct = 0
    total = 0
    
    optimizer.zero_grad()  # Zero gradients at the start
    
    batch_idx = 0
    iterator = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{total_epochs}]", unit="batch")
    for batch_idx, (images, labels, _) in enumerate(iterator):
        images, labels = images.to(device), labels.to(device)

        labels = labels.long()

        # HARD CHECKS
        if (labels < 0).any():
            bad = labels[labels < 0][:10].detach().cpu().tolist()
            raise ValueError(f"Found negative labels in TRAIN batch: {bad}")

        if (labels >= loss_fn.num_classes).any():
            bad = labels[labels >= loss_fn.num_classes][:10].detach().cpu().tolist()
            raise ValueError(f"Found out-of-range labels in TRAIN batch: {bad} (num_classes={loss_fn.num_classes})")
        
        embeddings = model(images)

        # NEW: get ArcFace logits used by the loss
        loss, logits = loss_fn(embeddings, labels, return_logits=True)

        # Normalize loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Only update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Track metrics (use unnormalized loss for logging)
        total_loss += loss.item() * accumulation_steps

        # Accuracy from ArcFace logits (same decision surface as training)
        with torch.no_grad():
            predictions = logits.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
        
        if (batch_idx + 1) % (10 * accumulation_steps) == 0:
            print(f"  Batch [{batch_idx+1}/{len(train_loader)}] "
                  f"Loss: {loss.item() * accumulation_steps:.4f}")
    
    # Handle remaining batches that don't fill accumulation_steps
    if (batch_idx + 1) % accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    avg_loss = total_loss / len(train_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


def evaluate_one_epoch(model, loss_fn, val_loader, device):
    """
    Evaluate on validation set
    """
    model.eval()
    loss_fn.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    
    iterator = tqdm(val_loader, desc="Validation", unit="batch")
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            mask = labels >= 0
            if mask.sum() == 0:
                continue

            images = images[mask]
            labels = labels[mask]

            # Forward pass
            embeddings = model(images)
            loss = loss_fn(embeddings, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            embeddings_norm = F.normalize(embeddings, p=2, dim=1)
            weight_norm = F.normalize(loss_fn.weight, p=2, dim=1)
            logits = F.linear(embeddings_norm, weight_norm)
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    avg_loss = total_loss / len(val_loader)
    avg_acc = 100 * correct / total
    
    return avg_loss, avg_acc


def predict_with_knn_sklearn(model, train_loader, test_loader, device, k=1, metric='cosine', verbose=True):
    """
    k-NN prediction using scikit-learn's KNeighborsClassifier
    
    Args:
        model: Trained backbone model
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
        device: 'cuda' or 'cpu'
        k: Number of neighbors (paper uses k=1)
        metric: Distance metric ('cosine' for ArcFace, as per paper)
        verbose: Print progress
    
    Returns:
        accuracy: Test accuracy percentage
        predictions: Predicted labels
        true_labels: Ground truth labels
    """
    model.eval()
    
    if verbose:
        print("\n" + "="*60)
        print("K-NN PREDICTION WITH SCIKIT-LEARN")
        print("="*60)
        print(f"k={k}, metric={metric}")
    
    # Extract training embeddings
    if verbose:
        print("\nExtracting training embeddings...")
    train_embeddings = []
    train_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            embeddings = model(images)
            
            # Normalize embeddings (important for cosine similarity)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            train_embeddings.append(embeddings.cpu().numpy())
            train_labels.append(labels.numpy())
            
            if verbose and (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx+1}/{len(train_loader)} batches")
    
    # Concatenate all batches
    train_embeddings = np.concatenate(train_embeddings, axis=0)
    train_labels = np.concatenate(train_labels, axis=0)
    
    if verbose:
        print(f"Training embeddings shape: {train_embeddings.shape}")
        print(f"Training labels shape: {train_labels.shape}")
    
    # Fit k-NN classifier
    if verbose:
        print(f"\nFitting k-NN classifier...")
    knn = KNeighborsClassifier(
        n_neighbors=k,
        metric=metric,
        algorithm='brute',  # Use brute force for cosine distance
        n_jobs=-1  # Use all CPU cores
    )
    knn.fit(train_embeddings, train_labels)
    
    # Extract test embeddings and predict
    if verbose:
        print("Extracting test embeddings and predicting...")
    test_embeddings = []
    test_labels = []
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            embeddings = model(images)
            
            # Normalize embeddings
            embeddings = F.normalize(embeddings, p=2, dim=1)
            
            test_embeddings.append(embeddings.cpu().numpy())
            test_labels.append(labels.numpy())
            
            if verbose and (batch_idx + 1) % 20 == 0:
                print(f"  Processed {batch_idx+1}/{len(test_loader)} batches")
    
    test_embeddings = np.concatenate(test_embeddings, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)
    
    if verbose:
        print(f"Test embeddings shape: {test_embeddings.shape}")
    
    # Predict
    predictions = knn.predict(test_embeddings)
    
    # Calculate accuracy
    correct = (predictions == test_labels).sum()
    total = len(test_labels)
    accuracy = 100 * correct / total
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"k-NN Results (k={k}):")
        print(f"  Accuracy: {accuracy:.2f}%")
        print(f"  Correct: {correct}/{total}")
        print(f"{'='*60}\n")
    
    return accuracy, predictions, test_labels


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