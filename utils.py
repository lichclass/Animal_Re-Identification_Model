import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os
import urllib.request
import zipfile
import json
import pandas as pd
import numpy as np
from PIL import Image
import ast
import matplotlib.pyplot as plt

# Function to download dataset
def download_dataset():
    data_dir = "data"
    dataset_dir = "turtle-data"
    
    # GitHub download URL for raw zip
    dataset_link = "https://github.com/lichclass/Animal_Re-Identification_Model/raw/main/downloads/turtle-data.zip"
    
    zip_path = os.path.join(data_dir, f"{dataset_dir}.zip")

    # Ensure root data directory exists
    if not os.path.exists(data_dir):
        print(f"Creating Data Directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    else:
        print(f'Data Directory "{data_dir}" already exists. Skipping creation...')

    # Skip if dataset already extracted
    dataset_path = os.path.join(data_dir, dataset_dir)
    if os.path.exists(dataset_path):
        print(f'Dataset Directory "{dataset_dir}" already exists. Skipping download...')
        return

    # Download zip file safely
    print(f"Downloading Dataset: {dataset_dir} from {dataset_link}")
    try:
        urllib.request.urlretrieve(dataset_link, zip_path)
        print("Download complete.")
    except Exception as e:
        print("Download FAILED:", e)
        return

    # Extract zip
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    except Exception as e:
        print("Extraction FAILED:", e)
        return
    finally:
        # Clean up: remove zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

    print("Dataset is ready!")

# By: Elijah
def build_sea_turtle_metadata(annotations, metadata, dataset_path):
    """
    Match bounding boxes from annotations to metadata according to category.
    Performs a left join to add bounding box information to metadata.
    
    Parameters:
    -----------
    annotations : str or dict
        Path to COCO format annotations JSON file or loaded annotations dict
    metadata : str or pd.DataFrame
        Path to metadata CSV file or loaded metadata DataFrame
    dataset_path : str
        Path to the dataset directory
    
    Returns:
    --------
    tuple of pd.DataFrame
        Three DataFrames for turtle, flipper, and head categories
    """

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

# By: Elijah
def crop_turtle(turtle_id, metadata_path, dataset_path="data/turtle-data"):
    """
    Crop images based on bounding boxes for a specific turtle identity.
    
    Parameters:
    -----------
    turtle_id : str
        The turtle identity to filter (e.g., 't001', 't002', etc.)
    metadata_path : str
        Path to the metadata CSV file with bounding boxes (e.g., metadata_splits_turtle.csv)
    dataset_path : str
        Path to the dataset directory containing images
    
    Returns:
    --------
    list of dict
        List of dictionaries containing:
        - 'cropped_image': PIL Image object of cropped region
        - 'file_name': original file name
        - 'bounding_box': bounding box coordinates [x, y, width, height]
        - 'category': category name
        - 'identity': turtle identity
    """
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

# By: Elijah
def display_cropped_images(cropped_results, max_display=10):
    """
    Display cropped images in a grid.
    
    Parameters:
    -----------
    cropped_results : list of dict
        Results from crop_turtle function
    max_display : int
        Maximum number of images to display
    """
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