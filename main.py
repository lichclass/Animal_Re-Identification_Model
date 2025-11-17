import argparse
from inspect_annot import inspect_annotations

parser = argparse.ArgumentParser(
    description='Federated Prototypical Network with Residual Networks and Atrous Spatial Pyramid Pooling for SeaTurtleID2022'
)

parser.add_argument(
    "-ia", "--inspect-annotations",
    action="store_true",
    help="Inspect the dataset annotations"
)

if __name__ == "__main__":
    args = parser.parse_args()

    if args.inspect_annotations:
        inspect_annotations()
