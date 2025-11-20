import json
import os

def inspect_annotations():

    # Configurations
    DATA_DIR = "data"
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "turtle-data", "annotations.json")
    sample_idx = 0 # the index of the sample to display

    # Read the JSON file
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    # For displaying information
    file_size = os.path.getsize(ANNOTATIONS_PATH)
    keys = list(annotations.keys())
    categories = [{'id': c['id'], 'name': c['name']} for c in annotations['categories']]
    sample_img = annotations["images"][sample_idx]
    sample_annotations = annotations["annotations"][sample_idx]

    print(f"""
=====================================================
SeaTurtleID2022 Dataset Annotations JSON Contents:
=====================================================

> JSON File Size: 
    {file_size} bytes
    {file_size / 1024} KB
    {file_size / 1024 / 1024} MB

> List of Keys: 
    {keys}

> List of Categories (count = {len(categories)}): 
    {categories}

> Sample Image Data (index: {sample_idx}): 
    {sample_img}

> Sample Annotation Data (index: {sample_idx}): 
    {sample_annotations}

=====================================================
    """)
