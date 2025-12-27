import torch
import torch.optim as optim
import pandas as pd
import numpy as np

from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import grad_scaler, autocast_mode
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import (
    extract_embeddings,
    compute_rank1_rank5_map,
    build_dataset_splits,
    build_model,
    plot_loss_curve,
    plot_tsne,
    set_seed
)


def main():
    
    split_mode = 'closed'
    segment = 'head'

    BACKBONE = 'convnext'
    LOSS_HEAD = 'adaface'

    DATASET_DIR = Path('/content/turtle-data')
    METADATA_FILE = DATASET_DIR / f'metadata_splits_{segment}.csv'
    RESULTS_DIR = Path("/content/results")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_NAME_DIR = RESULTS_DIR / f"{BACKBONE}_{LOSS_HEAD}_{split_mode}_{segment}"
    RESULTS_NAME_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH = RESULTS_NAME_DIR / "best_model.pth"
    TSNE_PLOTS_DIR = RESULTS_NAME_DIR / "tsne_plots"
    TSNE_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(METADATA_FILE)
    df['file_name'] = df['file_name'].apply(lambda x: DATASET_DIR / x)
    df = df[['file_name', 'identity', 'date', f'split_{split_mode}', 'bounding_box']]
    identities = df['identity'].unique().tolist()
    identity_to_index = {identity: idx for idx, identity in enumerate(identities)}
    index_to_identity = {idx: identity for identity, idx in identity_to_index.items()}
    df['label'] = df['identity'].map(identity_to_index)
    df['encounter_identity_date'] = df['identity'].astype(str) + "_" + df['date'].astype(str)
    unique_encounters = df['encounter_identity_date'].unique().tolist()
    encounter_id_to_index = {encounter: idx for idx, encounter in enumerate(unique_encounters)}
    index_to_encounter_id = {idx: encounter for encounter, idx in encounter_id_to_index.items()}
    df['encounter_label'] = df['encounter_identity_date'].map(encounter_id_to_index)

    # Final Columns: ['file_name', 'identity', 'date', f'split_{split_mode}', 'bounding_box', 'label', 'encounter_identity_date', 'encounter_label']

    train_set, val_set, test_set = build_dataset_splits(df, split_mode)
    print("Train/Val/Test sizes:", len(train_set), len(val_set), len(test_set))

    set_seed(42)

    EPOCHS = 50
    LEARNING_RATE = 1e-4

    if LOSS_HEAD == 'adaface':
        model = build_model(embedding_dim=512, num_classes=len(identities), backbone_type=BACKBONE, head_type=LOSS_HEAD, pretrained_backbone=True, dropout=0.1, m=0.4, h=0.333, s=64.0, t_alpha=1.0)
    elif LOSS_HEAD == 'arcface':
        model = build_model(embedding_dim=512, num_classes=len(identities), backbone_type=BACKBONE, head_type=LOSS_HEAD, pretrained_backbone=True, dropout=0.1, s=32.0, m=0.5)

    if MODEL_PATH.exists():
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f"Loaded model weights from {MODEL_PATH}")

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    model = model.to(DEVICE)

    CRITERION = torch.nn.CrossEntropyLoss()
    OPTIMIZER = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    SCHEDULER = CosineAnnealingLR(OPTIMIZER, T_max=EPOCHS, eta_min=1e-6)
    SCALER = grad_scaler.GradScaler()

    BATCH_SIZE = 64
    NUM_WORKERS = 2
    PIN_MEMORY = True
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    EARLY_STOPPING_PATIENCE = 5
    BEST_RANK1 = 0.0
    BEST_RANK5 = 0.0
    BEST_MAP = 0.0

    patience_counter = 0
    eval_every = 1

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rank1": [],
        "val_rank5": [],
        "val_map": [],
    }

    for epoch in range(1, EPOCHS + 1):

        # Training phase
        model.train()
        running_loss = 0.0
        iterator = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} - Training")
        for images, labels, _ in iterator:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            OPTIMIZER.zero_grad()
            with autocast_mode.autocast(device_type='cuda'):
                logits, _ = model(images, labels)
                loss = CRITERION(logits, labels)
            SCALER.scale(loss).backward()
            SCALER.step(OPTIMIZER)
            SCALER.update()
            running_loss += loss.item() * images.size(0)
            iterator.set_postfix(loss=loss.item())
        SCHEDULER.step()
        epoch_loss = running_loss / len(train_set)
        print(f"Epoch {epoch} Training Loss: {epoch_loss:.4f}")

        # Evaluation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            iterator = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} - Validation")
            for images, labels, _ in iterator:
                images = images.to(DEVICE, non_blocking=True)
                labels = labels.to(DEVICE, non_blocking=True)
                logits, _ = model(images, labels)
                loss = CRITERION(logits, labels)
                val_loss += loss.item() * images.size(0)
                iterator.set_postfix(loss=loss.item())
        val_loss /= len(val_set)
        print(f"Epoch {epoch} Validation Loss: {val_loss:.4f}")

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        # Perform Periodic Re-ID Evaluation
        do_eval = (epoch % eval_every == 0) or ((epoch + 1) % eval_every == 0)
        if do_eval:
            val_embs, val_labels, val_encounters = extract_embeddings(model, val_loader, DEVICE, set_name='Val')

            rank1, rank5, mAP = compute_rank1_rank5_map(
                val_embs, val_labels, val_encounters,
                val_embs, val_labels, val_encounters,
                device=DEVICE
            )
            print(f"Epoch {epoch} Validation Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%, mAP: {mAP*100:.2f}%")

            history["val_rank1"].append(rank1)
            history["val_rank5"].append(rank5)
            history["val_map"].append(mAP)

            if rank1 > BEST_RANK1:
                BEST_RANK1 = rank1
                BEST_RANK5 = rank5
                BEST_MAP = mAP
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"✅ New best model saved with Rank-1: {BEST_RANK1*100:.2f}%")

                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print("Early stopping triggered.")
                    break
    print(f"Training completed. Best Validation Rank-1: {BEST_RANK1*100:.2f}%, Rank-5: {BEST_RANK5*100:.2f}%, mAP: {BEST_MAP*100:.2f}%")

    plot_loss_curve(history, RESULTS_NAME_DIR / "loss_curve.png")

    # Final Testing Phase
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    test_embs, test_labels, test_encounters = extract_embeddings(model, test_loader, DEVICE, set_name='Test')
    rank1, rank5, mAP = compute_rank1_rank5_map(
        test_embs, test_labels, test_encounters,
        test_embs, test_labels, test_encounters,
        device=DEVICE
    )
    print(f"Final Test Set Results - Rank-1: {rank1*100:.2f}%, Rank-5: {rank5*100:.2f}%, mAP: {mAP*100:.2f}%")
    with open(RESULTS_NAME_DIR / "test_results.txt", "w") as f:
        f.write(f"Re-identification Test Set Results:\n")
        f.write(f"Rank-1: {rank1*100:.2f}%\n")
        f.write(f"Rank-5: {rank5*100:.2f}%\n")
        f.write(f"mAP: {mAP*100:.2f}%\n")

    emb, lab = test_embs.cpu().numpy(), test_labels.cpu().numpy()
    MAX_POINTS = 2000
    if emb.shape[0] > MAX_POINTS:
        idx = np.random.RandomState(42).choice(emb.shape[0], MAX_POINTS, replace=False)
        emb = emb[idx]
        lab = lab[idx]

    # Plot a T-SNE of the test embeddings
    plot_tsne(emb, lab, title="T-SNE of Test Embeddings", save_path=TSNE_PLOTS_DIR / "tsne_query_test.png")