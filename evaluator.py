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
def evaluate_one_epoch(model, eval_dataset, task_sampler, n_support, device="cuda"):
    model.to(device)
    model.eval()

    map1_scores = []
    map5_scores = [] 

    loss_scores = []
    acc_scores = []

    for batch_indices in tqdm(task_sampler, desc="Val episodes", leave=False):
        
        # Load one episodic batch
        batch = [eval_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Forward pass → embeddings
        embeddings = model(imgs)

        # Split into Support + Query
        (
            prototypes,
            query_embeddings,
            query_labels,
            class_to_idx,
            classes
        ) = split_support_query(embeddings, labels, n_support)

        # Compute distances
        distances = euclidean_distance(query_embeddings, prototypes)
        # shape: [num_queries, n_classes]

        # Compute loss and acc
        log_prob = torch.nn.functional.log_softmax(-distances, dim=1)
        loss = torch.nn.functional.nll_loss(log_prob, query_labels)
        loss_scores.append(loss.item())

        preds = torch.argmin(distances, dim=1)
        acc = (preds == query_labels).float().mean()
        acc_scores.append(acc.item())

        # Compute mAP metrics
        m1, m5 = compute_map(distances, query_labels, k=5)

        map1_scores.append(m1)
        map5_scores.append(m5)

    # Return episode-averaged mAP
    map1 = torch.tensor(map1_scores).mean().item()
    map5 = torch.tensor(map5_scores).mean().item()
    loss_avg = torch.tensor(loss_scores).mean().item()
    acc_avg = torch.tensor(acc_scores).mean().item()

    return map1, map5, loss_avg, acc_avg