import torch
import torch.nn.functional as F

from tqdm import tqdm
from utils.metrics import compute_map
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

        imgs = torch.stack(imgs)
        labels = torch.tensor(labels, dtype=torch.long)

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