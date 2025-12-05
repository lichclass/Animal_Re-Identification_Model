import os
import pandas as pd
import json


def inspect_annotations():
    DATA_DIR = "data"
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "turtle-data", "annotations.json")
    sample_idx = 0

    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

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


def inspect_metadata():
    DATA_DIR = "data"
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "turtle-data", "metadata_splits.csv")
    samples_len = 5

    data = pd.read_csv(ANNOTATIONS_PATH)

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

