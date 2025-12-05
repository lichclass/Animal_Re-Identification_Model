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