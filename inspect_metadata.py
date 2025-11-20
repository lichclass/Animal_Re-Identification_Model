import os
import pandas as pd

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
