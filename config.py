import argparse
import os
import yaml

def get_config():
    parser = argparse.ArgumentParser(
        description="Federated Prototypical Network with Residual Networks and ASPP for SeaTurtleID2022"
    )

    # Global parameter
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Utility actions
    parser.add_argument("--download-data", action="store_true", help="Download and extract the SeaTurtleID2022 dataset into ./data/")
    parser.add_argument("--build-splits", action="store_true", help="Build metadata splits (turtle, flipper, head) from annotations.json and metadata_splits.csv")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    # For Test suite
    parser.add_argument("--run-test", action="store_true", help="Run the test suite")
    parser.add_argument("--test-results-dir", type=str, help="Directory to save and retrieve test materials")
    parser.add_argument("--test-model-path", type=str, help="Path to a pretrained model for testing")
    parser.add_argument("--test-method", type=str, default=None, choices=[None, "major_vote", "emb_avg"], help="Method for encounter-level prediction during testing")

    # Data Inspection
    parser.add_argument("--inspect-annotations", action="store_true")
    parser.add_argument("--inspect-metadata", action="store_true")
   
    # Dataset parameters
    parser.add_argument("--dataset-dir", type=str, default="./data/turtle-data")
    parser.add_argument("--annotations", type=str, default="./data/turtle-data/annotations.json")
    parser.add_argument("--metadata", type=str, default="./data/turtle-data/metadata_splits.csv")
    parser.add_argument("--results-path", type=str, default="results")
    parser.add_argument("--split-mode", type=str, default="closed", help="closed, open") # closed, open
    parser.add_argument("--segment", type=str, default="head") # turtle, flipper, head, full
    
    # Model parameters
    parser.add_argument("--backbone", type=str, default="convnext", choices=["convnext", "swin"])
    parser.add_argument("--head", type=str, default="adaface", choices=["arcface", "adaface"])
    parser.add_argument("--embedding-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.2)

    # For ArcFace / AdaFace Head
    parser.add_argument("--s", type=float, default=64.0, help="Scaling factor for ArcFace/AdaFace")
    parser.add_argument("--m", type=float, default=0.35, help="Angular margin for ArcFace/AdaFace")

    # For AdaFace Head
    parser.add_argument("--h", type=float, default=0.2, help="Hardness sensitivity for AdaFace")
    parser.add_argument("--t-alpha", type=float, default=0.01, help="Temporal alpha for AdaFace")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs for each client/number of epochs for centralized training")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--early-stopping-patience", type=int, default=5, help="Early stopping patience based on evaluation accuracy")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--warmup-epochs", type=int, default=5, help="Number of warmup epochs for centralized training")
    parser.add_argument("--warmup-rounds", type=int, default=3, help="Number of warmup rounds for federated training")

    # Federation Configs
    parser.add_argument("--federated", action="store_true", help="Enable federated training")
    parser.add_argument("--num-clients", type=int, default=5, help="Number of federated clients")
    parser.add_argument("--federated-rounds", type=int, default=10, help="Number of federated training rounds")
    parser.add_argument("--lambda-proto", type=float, default=1.0, help="Weight for prototypical loss in federated training")
    parser.add_argument("--proto-momentum", type=float, default=0.9, help="Momentum for updating global prototypes in federated training")
    parser.add_argument("--overlap-ratio", type=float, default=0.1, help="Overlap ratio of identities between clients in federated training")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N federated/centralized rounds/epochs")

    # Dataloader Configs
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training and evaluation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--pin-memory", action="store_true", help="Pin memory for data loaders")

    args = parser.parse_args()
    return args


def save_config(args):
    results_path = os.path.join(args.results_dir, args.experiment_name)
    os.makedirs(results_path, exist_ok=True)
    cfg_path = os.path.join(results_path, "config.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(vars(args), f)
    return results_path
