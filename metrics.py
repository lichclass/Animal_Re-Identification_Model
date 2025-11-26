# ------------------------------------------------------------
# File Name: metrics.py
# Status: 🔧 READY FOR TESTING
# Revised: November 26, 2025
# Revised by: Nash Adam Muñoz
# File Description: 
#    This file contains the code for the evaluation metrics in 
#    in evaluating the model.
#
# Changes:
# - (November 26, 2025) changed the compute_map function to be
#   more flexible from top1 and top5 accuracy only to 
#   top1 and topk accuracy
# ------------------------------------------------------------

import torch

def compute_map(distances, labels, k=5):
    """
    distances: shape [num_queries, num_classes]
    labels: ground truth class index per query

    Args:
        distances: torch tensor of shape [num_queries, num_classes]
        labels: torch tensor of shape [num_queries]
        k: for mAP@k

    Returns:
        mAP@1, mAP@k
    """
    ranks = distances.argsort(dim=1)   # [N, C]
    
    # top-1 accuracy
    top1 = (ranks[:, 0] == labels).float().mean().item()
    
    # top-k accuracy (vectorized)
    # ranks[:, :k] → [N, k]
    # labels.unsqueeze(1) → [N, 1]
    topk = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    return top1, topk