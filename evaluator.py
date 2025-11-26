# ------------------------------------------------------------
# File Name: evaluator.py
# Status: 🔧 READY FOR TESTING
# Revised: November 26, 2025
# Revised by: Nash Adam Muñoz
# File Description: 
#    This file contains the code for evaluating the model.
# ------------------------------------------------------------

import torch
from tqdm import tqdm

from metrics import compute_map
from modules.protonetloss import (
    split_support_query,
    euclidean_distance,
)

@torch.no_grad()
def evaluate_one_epoch(model, eval_dataset, task_sampler, loss_fn, device="cuda"):
    model.to(device)
    model.eval()

    map1_scores = []
    map5_scores = []

    n_support = loss_fn.n_support   # read once

    for batch_indices in tqdm(task_sampler, desc="Val episodes", leave=False):
        
        # -----------------------------
        # Load one episodic batch
        # -----------------------------
        batch = [eval_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # -----------------------------
        # Forward pass → embeddings
        # -----------------------------
        embeddings = model(imgs)

        # -----------------------------
        # Split into Support + Query
        # (modular → reuses your logic)
        # -----------------------------
        (
            prototypes,
            query_embeddings,
            query_labels,
            class_to_idx,
            classes
        ) = split_support_query(embeddings, labels, n_support)

        # -----------------------------
        # Compute distances
        # -----------------------------
        distances = euclidean_distance(query_embeddings, prototypes)
        # shape: [num_queries, n_classes]

        # -----------------------------
        # Compute mAP metrics
        # -----------------------------
        m1, m5 = compute_map(distances, query_labels, k=5)

        map1_scores.append(m1)
        map5_scores.append(m5)

    # -----------------------------
    # Return episode-averaged mAP
    # -----------------------------
    map1 = torch.tensor(map1_scores).mean().item()
    map5 = torch.tensor(map5_scores).mean().item()

    return map1, map5