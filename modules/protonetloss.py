# All thanks to orobix for the code implementation of Snell et al's (2017)
# Prototypical Networks
# Github Repo: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class PrototypicalLoss(nn.Module):
    def __init__(self, n_support):
        super().__init__()
        self.n_support = n_support

    def forward(self, embeddings, labels):
        return prototypical_loss(embeddings, labels, self.n_support)


def euclidean_distance(x, y):
    # x: N x D
    # y: M x D
    
    N = x.size(0)
    M = y.size(0)
    D = x.size(1)
    
    assert D == y.size(1), "The second dimension of both tensors must be the same!"

    x = x.unsqueeze(1).expand(N, M, D)
    y = y.unsqueeze(0).expand(N, M, D)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(embeddings, labels, n_support):
    """
    Args:
        embeddings: output of the encoder (batch_size, embedding_dim)
        labels: ground-truth class labels for each embedding
        n_support: num of support examples per class
    
    Return:
        loss, accuracy
    """

    embeddings = embeddings.cpu()
    labels = labels.cpu()

    classes = torch.unique(labels)
    n_classes = len(classes)

    def get_support_indices(c):
        class_indices = (labels == c).nonzero(as_tuple=False).squeeze(1)
        return class_indices[:n_support]

    # Get support indices and compute prototypes
    support_indices = [get_support_indices(c) for c in classes]
    prototypes = torch.stack([embeddings[idxs].mean(dim=0) for idxs in support_indices], dim=0)

    # Get query indices for each class and concatenate them
    query_indices_list = []
    for c in classes:
        class_indices = (labels == c).nonzero(as_tuple=False).squeeze(1)
        class_query_indices = class_indices[n_support:]
        query_indices_list.append(class_query_indices)
    
    # Concatenate all query indices
    query_indices = torch.cat(query_indices_list, dim=0)

    # Get query embeddings and labels
    query_embeddings = embeddings[query_indices]
    query_labels = labels[query_indices]
    
    # Compute distances between query embeddings and prototypes
    distances = euclidean_distance(query_embeddings, prototypes)

    # Compute log probabilities
    log_probabilities = F.log_softmax(-distances, dim=1)
    
    # Create target labels that map to class indices (0, 1, 2, ..., n_classes-1)
    # Map original class labels to indices
    class_to_idx = {c.item(): idx for idx, c in enumerate(classes)}
    target_labels = torch.tensor([class_to_idx[label.item()] for label in query_labels])
    
    # Compute loss
    loss = F.nll_loss(log_probabilities, target_labels)
    
    # Compute accuracy
    predicted = log_probabilities.argmax(dim=1)
    accuracy = (predicted == target_labels).float().mean()
    
    return loss, accuracy