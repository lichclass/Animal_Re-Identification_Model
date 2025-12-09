# ------------------------------------------------------------
# File Name: protonetloss.py
# Status: 🔧 READY FOR TESTING
# Revised: November 26, 2025
# Revised by: Nash Adam Muñoz
# File Description: 
#    This file contains the code for the prototypical loss in
#    in the Prototypical Network.
#
# Changes:
# - (November 26, 2025) Modularized certain chunks of code.
# ------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------------------------------------------
# Prototypical Loss Class Wrapper
# -------------------------------------------------------------
class PrototypicalLoss(nn.Module):
    def __init__(self, n_support):
        super().__init__()
        self.n_support = n_support

    def forward(self, embeddings, labels):
        return prototypical_loss(embeddings, labels, self.n_support)


# -------------------------------------------------------------
# Euclidean Distance
# -------------------------------------------------------------
def euclidean_distance(x, y):
    # x: N x D
    # y: M x D
    
    N = x.size(0)
    M = y.size(0)
    D = x.size(1)
    
    assert D == y.size(1), "The second dimension of both tensors must be the same!"

    x = x.unsqueeze(1).expand(N, M, D)
    y = y.unsqueeze(0).expand(N, M, D)

    return torch.pow(x - y, 2).sum(2)   # [N, M]


# Compute Prototypes
def compute_prototypes(embeddings, labels, n_support):
    classes = torch.unique(labels)
    prototypes = []

    for c in classes:
        idx = (labels == c).nonzero(as_tuple=False).squeeze(1)
        support_idx = idx[:n_support]
        proto = embeddings[support_idx].mean(dim=0)
        prototypes.append(proto)

    return torch.stack(prototypes, dim=0), classes


# Split Support/Query 
def split_support_query(embeddings, labels, n_support):
    device = embeddings.device  

    # Work with labels on CPU for safer indexing
    labels_cpu = labels.detach().to("cpu")

    # Prototypes still computed as before
    prototypes, classes = compute_prototypes(embeddings, labels, n_support)

    # Map labels to episodic class indices
    class_to_idx = {c.item(): i for i, c in enumerate(classes)}

    # Build query indices per class
    q_indices = []
    for c in classes:
        idxs = (labels_cpu == c.item()).nonzero(as_tuple=False).view(-1)
        if idxs.numel() > n_support:
            q_indices.append(idxs[n_support:])

    if len(q_indices) == 0:
        raise ValueError("No query samples in episode; check n_support / sampler.")

    q_indices = torch.cat(q_indices, dim=0).to(device)

    # Index on device tensors
    q_emb = embeddings.index_select(0, q_indices)
    q_lbl_orig = labels.index_select(0, q_indices)

    # Remap labels to episodic class indices on CPU, then move back
    q_lbl = torch.tensor(
        [class_to_idx[int(l.item())] for l in q_lbl_orig.detach().to("cpu")],
        device=device,
        dtype=torch.long,
    )

    return prototypes, q_emb, q_lbl, class_to_idx, classes


# -------------------------------------------------------------
# Compute ProtoNet Loss
# -------------------------------------------------------------
def compute_protonet_loss(prototypes, query_embeddings, query_labels):
    distances = euclidean_distance(query_embeddings, prototypes)
    log_probs = F.log_softmax(-distances, dim=1)

    loss = F.nll_loss(log_probs, query_labels)

    pred = torch.argmax(log_probs, dim=1)
    acc = (pred == query_labels).float().mean()

    return loss, acc


# -------------------------------------------------------------
# Prototypical Loss Wrapper
# -------------------------------------------------------------
def prototypical_loss(embeddings, labels, n_support):
    (
        prototypes,
        query_embeddings,
        query_labels,
        class_to_idx,
        classes
    ) = split_support_query(embeddings, labels, n_support)

    loss, acc = \
        compute_protonet_loss(
            prototypes, 
            query_embeddings, 
            query_labels
        )

    return loss, acc


# -------------------------------------------------------------
# Global Prototype Alignment Loss (FedProto)
# -------------------------------------------------------------
def compute_alignment_loss(
    local_prototypes: torch.Tensor,
    local_classes: torch.Tensor,
    global_prototypes: torch.Tensor,
    global_classes: torch.Tensor,
) -> torch.Tensor:
    """
    Compute alignment loss that pulls local prototypes toward global prototypes.
    
    Args:
        local_prototypes: [n_local_classes, D] - local prototypes from current episode
        local_classes: [n_local_classes] - class IDs for local prototypes
        global_prototypes: [n_global_classes, D] - global prototypes from server
        global_classes: [n_global_classes] - class IDs for global prototypes
    
    Returns:
        alignment_loss: scalar tensor
    """
    if global_prototypes is None or len(global_prototypes) == 0:
        return torch.tensor(0.0, device=local_prototypes.device)
    
    # Create mapping from class ID to global prototype index
    global_class_to_idx = {
        cls.item() if isinstance(cls, torch.Tensor) else cls: idx
        for idx, cls in enumerate(global_classes)
    }
    
    alignment_losses = []
    
    # For each local prototype, find matching global prototype and compute distance
    for local_idx, local_class in enumerate(local_classes):
        local_class_item = local_class.item() if isinstance(local_class, torch.Tensor) else local_class
        
        # Check if this class exists in global prototypes
        if local_class_item in global_class_to_idx:
            global_idx = global_class_to_idx[local_class_item]
            local_proto = local_prototypes[local_idx]
            global_proto = global_prototypes[global_idx]
            
            # L2 distance between local and global prototype
            distance = torch.norm(local_proto - global_proto, p=2)
            alignment_losses.append(distance)
    
    if len(alignment_losses) == 0:
        return torch.tensor(0.0, device=local_prototypes.device)
    
    # Average alignment loss over matching classes
    return torch.stack(alignment_losses).mean()
