import os
import pandas as pd

def inspect_metadata():
    DATA_DIR = "data"
    ANNOTATIONS_PATH = os.path.join(DATA_DIR, "turtle-data", "annotations.json")
    with open(ANNOTATIONS_PATH, "r") as f:
        annotations = json.load(f)

    file_size = os.path.getsize(ANNOTATIONS_PATH)

    keys = list(annotations.keys())
    categories = [{'id': c['id'], 'name': c['name']} for c in annotations['categories']]

    sample_idx = 0
    sample_img = annotations["images"][sample_idx]
    sample_annotations = annotations["annotations"][sample_idx]

    print(f"""
=====================================================
SeaTurtleID2022 Metadata Information
=====================================================


=====================================================
    """)
