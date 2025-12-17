def compute_map(distances, labels, k=5):
    ranks = distances.argsort(dim=1)   # [N, C]
    
    top1 = (ranks[:, 0] == labels).float().mean().item()
    topk = (ranks[:, :k] == labels.unsqueeze(1)).any(dim=1).float().mean().item()
    
    return top1, topk