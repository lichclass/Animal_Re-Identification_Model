import argparse
from inspect_annotations import inspect_annotations
from inspect_metadata import inspect_metadata
from utils import download_dataset

parser = argparse.ArgumentParser(description='Federated Prototypical Network with Residual Networks and Atrous Spatial Pyramid Pooling for SeaTurtleID2022')
parser.add_argument("-ia", "--inspect-annotations", action="store_true", help="Inspect the dataset annotations") # Inspect the dataset annotations
parser.add_argument("-im", "--inspect-metadata", action="store_true", help="Inspect the dataset metadata")
parser.add_argument("-ov", "--overview", action="store_true", help="Display Project Overview") # Display the overview of the Thesis Project
parser.add_argument("-dd", "--download-dataset", action="store_true", help="Download the dataset")

parser.add_argument("-mes", "--message", action="store_true", help="Message from the developers") # Small Easter Egg
parser.add_argument("-kie", "--kie-message", action="store_true", help="frankie's message")

if __name__ == "__main__":
    args = parser.parse_args()

    # ================================================================
    # Descriptive Commands
    # ================================================================
    if args.overview:
        overview_msg = "Federated Prototypical Network with Residual Networks and Atrous Spatial Pyramid Pooling for SeaTurtleID2022"
        print("="*len(overview_msg))
        print("Project Overview: \n")
        print(overview_msg)
        print("="*len(overview_msg))

    if args.inspect_annotations:
        inspect_annotations()

    if args.inspect_metadata:
        inspect_metadata()

    # ================================================================
    # Utility Commands
    # ================================================================
    if args.download_dataset:
        download_dataset()


    # ================================================================
    # Miscellaneous commands
    # ================================================================
    if args.kie_message:
        print("u dont know how to update properly and you dont know how to clean.f")

    if args.message:
        developer_msg = "Nice to meet you! We are Nash Adam Muñoz and Elijah Kahlil Andres Abangan. Thank you for stopping by our Thesis Project!"
        print("="*len(developer_msg))
        print("Message from the Developers: \n")
        print(developer_msg)
        print("="*len(developer_msg))

