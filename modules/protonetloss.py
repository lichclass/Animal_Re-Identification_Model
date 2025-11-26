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
# Class: Prototypical Loss Class Wrapper
# -------------------------------------------------------------
class PrototypicalLoss(nn.Module):
    def __init__(self, n_support):
        super().__init__()
        self.n_support = n_support

    def forward(self, embeddings, labels):
        return prototypical_loss(embeddings, labels, self.n_support)


# -------------------------------------------------------------
# Function: Euclidean Distance
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


# -------------------------------------------------------------
# Function: Split Embeddings into Support and Query sets
# -------------------------------------------------------------
def split_support_query(embeddings, labels, n_support):
    embeddings = embeddings.cpu()
    labels = labels.cpu()

    classes = torch.unique(labels)
    class_to_idx = {c.item(): idx for idx, c in enumerate(classes)}

    # Getting the Support Set
    support_indices = []
    for c in classes:
        class_indices = (labels == c).nonzero(as_tuple=False).squeeze(1)
        support_indices.append(class_indices[:n_support])

    # Getting the prototypes
    prototypes = torch.stack(
        [embeddings[idxs].mean(dim=0) for idxs in support_indices],
        dim=0
    )  # [n_classes, D]

    # Getting the Query Set
    query_indices_list = []
    for c in classes:
        class_indices = (labels == c).nonzero(as_tuple=False).squeeze(1)
        class_query_indices = class_indices[n_support:]
        query_indices_list.append(class_query_indices)

    query_indices = torch.cat(query_indices_list, dim=0)
    query_embeddings = embeddings[query_indices]   # [num_queries, D]
    query_labels = labels[query_indices]           # [num_queries]

    mapped_query_labels = torch.tensor(
        [class_to_idx[label.item()] for label in query_labels]
    )

    return (
        prototypes,
        query_embeddings,
        mapped_query_labels,
        class_to_idx,
        classes
    )


# -------------------------------------------------------------
# Function: Compute ProtoNet Loss
# -------------------------------------------------------------
def compute_protonet_loss(prototypes, query_embeddings, query_labels):
    # queries vs prototypes → [num_queries, n_classes]
    distances = euclidean_distance(query_embeddings, prototypes)

    log_probs = F.log_softmax(-distances, dim=1)   # softmax over classes
    loss = F.nll_loss(log_probs, query_labels)

    preds = log_probs.argmax(dim=1)
    acc = (preds == query_labels).float().mean()

    return loss, acc, distances


# -------------------------------------------------------------
# Function: Prototypical Loss Pipeline
# -------------------------------------------------------------
def prototypical_loss(embeddings, labels, n_support):
    """
    Wrapper function for all the necessary steps to compute the Prototypical Loss
    Steps:
        1. Split embeddings into support and query sets
        2. Compute prototypes
        3. Compute distances + NLL loss
        4. Compute accuracy
    """ 
    (
        prototypes, 
        query_embeddings, 
        query_labels, 
        class_to_idx, 
        classes
    ) = split_support_query(embeddings, labels, n_support) 

    loss, acc, distances = compute_protonet_loss(
        prototypes, query_embeddings, query_labels
    )

    return loss, acc
