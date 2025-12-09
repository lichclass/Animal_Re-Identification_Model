import argparse
import os
import yaml
import torch

def get_config():
    parser = argparse.ArgumentParser(
        description="Federated Prototypical Network with Residual Networks and ASPP for SeaTurtleID2022"
    )

    # Action Commands
    parser.add_argument(
        "--download-data",
        action="store_true",
        help="Download and extract the SeaTurtleID2022 dataset into ./data/"
    )

    parser.add_argument(
        "--build-splits",
        action="store_true",
        help="Build metadata splits (turtle, flipper, head) from annotations.json and metadata_splits.csv"
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train the model"
    )

    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default="data/turtle-data")
    parser.add_argument("--annotations", type=str, default="data/turtle-data/annotations.json")
    parser.add_argument("--metadata", type=str, default="data/turtle-data/metadata_splits.csv")
    parser.add_argument("--segment", type=str, default="turtle") # turtle, flipper, head
    parser.add_argument("--split-mode", type=str, default="closed", help="closed, open, random") # closed, open, random

    # Task sampling parameters
    parser.add_argument("--n-way", type=int, default=5)
    parser.add_argument("--k-shot", type=int, default=5)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--eval-episodes", type=int, default=100)
    parser.add_argument("--test-episodes", type=int, default=100)
    
    # Federated Prototypical Network parameters
    parser.add_argument("--federated", action="store_true", help="Use Federated Prototypical Network")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--lambda-align", type=float, default=0.5, help="Weight for prototype alignment loss")
    parser.add_argument("--lambda-triplet", type=float, default=0.3, help="Weight for triplet loss")
    parser.add_argument("--num-clients", type=int, default=3)

    # Model parameters
    parser.add_argument(
        "--backbone",
        type=str,
        choices=["resnet18", "resnet18_aspp"],
        default="resnet18_aspp"
    )
    parser.add_argument("--embedding-dim", type=int, default=256)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--early-stopping-patience", type=int, default=10)

    # Output and Logging
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--experiment-name", type=str, default="exp1")

    # Data Inspection
    parser.add_argument("--inspect-annotations", action="store_true")
    parser.add_argument("--inspect-metadata", action="store_true")

    args = parser.parse_args()
    return args


def save_config(args):
    results_path = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(results_path, exist_ok=True)
    cfg_path = os.path.join(results_path, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(vars(args), f)
    return results_path
