import torch
import torch.optim as optim
import pandas as pd
import numpy as np
import os

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import grad_scaler, autocast_mode
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import get_config
from utils import (
    extract_embeddings,
    compute_rank1_rank5_map,
    build_dataset_splits,
    build_model,
    plot_tsne,
    set_seed,
    prep_dataframe,
    build_sea_turtle_metadata,
    download_dataset,
    inspect_annotations,
    inspect_metadata,
)

def main():
    args = get_config()
    set_seed(args.seed)

    if args.download_data:
        download_dataset()
        return
    
    if args.inspect_annotations: inspect_annotations(); return
    if args.inspect_metadata: inspect_metadata(); return    

    # Build metadata splits
    if args.build_splits:
        assert os.path.exists(args.annotations), f"Annotations not found: {args.annotations}"
        assert os.path.exists(args.metadata), f"Metadata CSV not found: {args.metadata}"

        build_sea_turtle_metadata(
            annotations=args.annotations,
            metadata=args.metadata,
            dataset_path=args.dataset_dir
        )
        return

    # Run tests
    if args.run_test:
        df = prep_dataframe(args)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        if args.federated:
            pass
        else:
            train_df_raw = df[df[f'split_{args.split_mode}'] == 'train']
            train_identities = sorted(train_df_raw['identity'].unique().tolist())
            train_id_to_idx = {identity: idx for idx, identity in enumerate(train_identities)}
            df['train_label'] = df['identity'].map(train_id_to_idx)

            _, _, test_set = build_dataset_splits(df, args.split_mode)
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=args.pin_memory, persistent_workers=True)
            head_kwargs = {}
            if args.head == 'arcface':
                head_kwargs = {'s': args.s, 'm': args.m}
            elif args.head == 'adaface':
                head_kwargs = {'m': args.m, 'h': args.h, 's': args.s, 't_alpha': args.t_alpha}

            model = build_model(
                embedding_dim=args.embedding_dim,
                num_classes=len(train_identities),
                backbone_type=args.backbone,
                head_type=args.head,
                dropout=args.dropout,
                **head_kwargs
            ).to(device)

            model_path = Path(args.test_model_path)
            assert model_path.is_file(), f"Model path {model_path} does not exist."
            model.load_state_dict(torch.load(model_path, map_location=device))

            test_embs, test_labels, test_encounters = extract_embeddings(model, test_loader, device, set_name='Test')

            rank1, rank5, mAP = compute_rank1_rank5_map(
                test_embs, test_labels, test_encounters,
                test_embs, test_labels, test_encounters,
                device=device,
                encounter_based=args.test_method,
            )

            test_method = {
                None: "Image-Level",
                "major_vote": "Encounter-Level (Majority Vote)",
                "emb_avg": "Encounter-Level (Embedding Average)"
            }

            test_results_dir = Path(args.test_results_dir)
            test_results_dir.mkdir(parents=True, exist_ok=True)
            print(f"Test Results - Rank-1: {rank1*100:.2f}, Rank-5: {rank5*100:.2f}, mAP: {mAP*100:.2f}s")
            with open(test_results_dir / "test_results.txt", "w") as f:
                f.write(f"Test Results using {test_method[args.test_method]}:")
                f.write(f" Rank-1: {rank1*100:.2f}")
                f.write(f" Rank-5: {rank5*100:.2f}")
                f.write(f" mAP: {mAP*100:.2f}\n")
                f.write("\nConfig:\n")
                for k, v in vars(args).items():
                    f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()


        



