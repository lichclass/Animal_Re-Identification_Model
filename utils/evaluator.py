import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from tqdm import tqdm
from modules.protonetloss import (
    split_support_query,
    euclidean_distance,
)

@torch.no_grad()
def evaluate_one(model, eval_dataset, task_sampler, n_support, device="cuda"):
    model.to(device)
    model.eval()

    running_loss = []
    running_acc = []

    for batch_indices in tqdm(task_sampler, desc="Val episodes", leave=False):
        
        # Load one episodic batch
        batch = [eval_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # Forward pass → embeddings
        embeddings = model(imgs)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # Split into Support + Query
        (
            prototypes,
            query_embeddings,
            query_labels,
            _,
            _
        ) = split_support_query(embeddings, labels, n_support)

        # Compute distances
        distances = euclidean_distance(query_embeddings, prototypes)
        # shape: [num_queries, n_classes]

        # Compute loss and acc
        log_prob = torch.nn.functional.log_softmax(-distances, dim=1)
        loss = torch.nn.functional.nll_loss(log_prob, query_labels)
        running_loss.append(loss.item())

        preds = torch.argmin(distances, dim=1)
        acc = (preds == query_labels).float().mean()
        running_acc.append(acc.item())

    loss = torch.tensor(running_loss).mean().item()
    acc = torch.tensor(running_acc).mean().item()

    return loss, acc


@torch.no_grad()
def evaluate_reid(model, query_dataset, gallery_dataset, batch_size=64, device=None, topk=5):
    device = device or torch.device("cpu")
    model.to(device).eval()

    def extract_features(dataset):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=(device.type == "cuda"),
        )
        feats, labs = [], []
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            emb = model(imgs)
            emb = F.normalize(emb, p=2, dim=1)
            feats.append(emb)
            labs.append(labels.to(device))
        if not feats:
            return None, None
        return torch.cat(feats, dim=0), torch.cat(labs, dim=0)

    q_feats, q_labels = extract_features(query_dataset)
    g_feats, g_labels = extract_features(gallery_dataset)

    if q_feats is None or g_feats is None:
        return 0.0, 0.0, 0.0

    dist_matrix = torch.cdist(q_feats, g_feats, p=2)  # [Nq, Ng]

    # sort once
    indices = torch.argsort(dist_matrix, dim=1)       # [Nq, Ng]

    mAP = compute_map_from_indices(indices, q_labels, g_labels)
    cmc = compute_cmc_from_indices(indices, q_labels, g_labels, topk)

    rank1 = cmc[0].item()
    rank5 = cmc[min(4, len(cmc)-1)].item()

    return float(mAP), rank1, rank5
    

def compute_map_from_indices(indices, q_labels, g_labels):
    """
    Compute mean Average Precision (mAP) for ReID using precomputed ranking indices.

    Args:
        indices: [Nq, Ng] LongTensor
            indices[i] is the ranking of gallery indices for query i (ascending distance).
        q_labels: [Nq] LongTensor
            Identity labels for queries.
        g_labels: [Ng] LongTensor
            Identity labels for gallery images.

    Returns:
        mAP: scalar tensor with the mean AP over all queries that have at least one match.
    """
    device = q_labels.device
    num_q = indices.size(0)

    APs = []

    for i in range(num_q):
        q_label = q_labels[i]

        # Gallery labels sorted by distance for this query
        ranked_g_labels = g_labels[indices[i]]  # [Ng]

        # Binary relevance vector: 1 if same identity, 0 otherwise
        matches = (ranked_g_labels == q_label).to(torch.float32)  # [Ng]

        num_rel = matches.sum()
        if num_rel == 0:
            # No relevant gallery images for this query; skip it (common in ReID eval)
            continue

        # Positions of correct matches (0-based ranks)
        correct_idx = matches.nonzero(as_tuple=False).squeeze(1)  # [R], R = num_rel

        # Precision@k for those positions:
        # precision(k) = (# relevant in top-k) / k
        # cumulative sum of matches gives #relevant up to each rank
        cum_matches = torch.cumsum(matches, dim=0)                # [Ng]
        ranks = correct_idx + 1                                   # convert to 1-based ranks
        precision_at_k = cum_matches[correct_idx] / ranks.to(device)

        # Average Precision for this query = mean precision at all relevant positions
        AP = precision_at_k.mean()
        APs.append(AP)

    if len(APs) == 0:
        return torch.tensor(0.0, device=device)

    mAP = torch.stack(APs).mean()
    return mAP


def compute_cmc_from_indices(indices, q_labels, g_labels, max_rank=5):
    """
    Compute Cumulative Matching Characteristic (CMC) curve up to `max_rank`
    using precomputed ranking indices.

    Args:
        indices: [Nq, Ng] LongTensor
            indices[i] is the ranking of gallery indices for query i (ascending distance).
        q_labels: [Nq] LongTensor
            Identity labels for queries.
        g_labels: [Ng] LongTensor
            Identity labels for gallery images.
        max_rank: int
            Highest rank to compute CMC for (e.g., 5 → Rank-1..Rank-5).

    Returns:
        cmc: [max_rank] tensor
            cmc[k] = probability (over queries) that the correct match
            appears at rank ≤ k+1.
    """
    device = q_labels.device
    num_q = indices.size(0)

    # In case max_rank > number of gallery images, clamp it
    max_rank = min(max_rank, indices.size(1))

    cmc = torch.zeros(max_rank, device=device)

    for i in range(num_q):
        q_label = q_labels[i]

        # Ranked gallery labels for this query
        ranked_g_labels = g_labels[indices[i]]  # [Ng]

        # Positions where gallery identity matches query identity
        matches = (ranked_g_labels == q_label)

        if not matches.any():
            # No correct match for this query in the gallery
            continue

        # First correct match rank (0-based)
        first_match_rank = matches.nonzero(as_tuple=False)[0, 0].item()

        if first_match_rank < max_rank:
            # If first match at rank r, then ranks r..max_rank-1 are all "success" for this query
            cmc[first_match_rank:] += 1

    if num_q == 0:
        return cmc  # all zeros

    cmc = cmc / num_q
    return cmc




