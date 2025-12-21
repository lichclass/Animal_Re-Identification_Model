import argparse
import os
import yaml

def get_config():
    parser = argparse.ArgumentParser(
        description="Federated Prototypical Network with Residual Networks and ASPP for SeaTurtleID2022"
    )

    # Utility actions
    parser.add_argument("--download-data", action="store_true", help="Download and extract the SeaTurtleID2022 dataset into ./data/")
    parser.add_argument("--build-splits", action="store_true", help="Build metadata splits (turtle, flipper, head) from annotations.json and metadata_splits.csv")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default="data/turtle-data")
    parser.add_argument("--annotations", type=str, default="data/turtle-data/annotations.json")
    parser.add_argument("--metadata", type=str, default="data/turtle-data/metadata_splits.csv")
    parser.add_argument("--segment", type=str, default="turtle") # turtle, flipper, head
    parser.add_argument("--split-mode", type=str, default="closed", help="closed, open, random") # closed, open, random
    
    # Federated Network parameters
    parser.add_argument("--federated", action="store_true", help="Use Federated Prototypical Network")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--num-clients", type=int, default=3)

    # Model parameters
    parser.add_argument("--embedding-dim", type=int, default=256)

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--optimizer", type=str, default="sgd", choices=["adam", "sgd"])
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--early-threshold", type=float, default=0.5)

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
