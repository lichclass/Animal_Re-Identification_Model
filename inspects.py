import json
import os
import pandas as pd


# -------------------------------------------------------------
# Function: display annotation information
# -------------------------------------------------------------
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


# -------------------------------------------------------------
# Function: display metadata information
# -------------------------------------------------------------
def inspect_metadata():

    # Configurations
    DATA_DIR = "data"
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "turtle-data", "metadata_splits.csv")
    samples_len = 5 # the number of samples to display

    # Read the CSV file
    data = pd.read_csv(ANNOTATIONS_PATH)

    # For displaying information
    file_size = os.path.getsize(ANNOTATIONS_PATH)
    columns = list(data.columns)
    num_of_data = len(data)

    print(f"""
=====================================================
SeaTurtleID2022 Metadata Information
=====================================================

> CSV File Size:
    {file_size} bytes
    {file_size / 1024} KB
    {file_size / 1024 / 1024} MB

> Table Columns:
    {columns}

> Number of Data:
    {num_of_data}

> Sample Data:
{data.head(samples_len)}

=====================================================
    """)