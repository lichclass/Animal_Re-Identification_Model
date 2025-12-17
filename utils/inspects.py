import os
from pathlib import Path
import pandas as pd
import json


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

