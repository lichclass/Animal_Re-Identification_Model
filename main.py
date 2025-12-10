# main.py

import os
from config import get_config, save_config

# Dataset utilities
from utils.utils import (
    download_dataset,
    build_sea_turtle_metadata,
)

from utils.inspects import (
    inspect_annotations,
    inspect_metadata
)

from experiment import (
    without_federation,
    with_federation
)

def main():
    args = get_config()
    save_config(args)

    # =========================================================
    # ACTION 1: Download dataset
    # =========================================================
    if args.download_data:
        download_dataset()
        return

    # =========================================================
    # ACTION 2: Build metadata splits
    # =========================================================
    if args.build_splits:
        assert os.path.exists(args.annotations), f"Annotations not found: {args.annotations}"
        assert os.path.exists(args.metadata), f"Metadata CSV not found: {args.metadata}"

        build_sea_turtle_metadata(
            annotations=args.annotations,
            metadata=args.metadata,
            dataset_path=args.data_dir
        )
        return

    # =========================================================
    # ACTION 3: Inspection Tools
    # =========================================================
    if args.inspect_annotations:
        inspect_annotations()
        return

    if args.inspect_metadata:
        inspect_metadata()
        return

    # =========================================================
    # ACTION 4: Training and Evaluation Pipelines
    # =========================================================

    if args.train:
        if args.federated: 
            print("\nRunning Federated Prototypical Network...\n")
            with_federation(args, verbose=True)
        else:
            print("\nRunning Non-Federated Prototypical Network...\n")
            without_federation(args, verbose=True) # Non-federated (For ablation studies)
        return

if __name__ == "__main__":
    main()