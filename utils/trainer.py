import torch
import torch.nn.functional as F

from tqdm import tqdm
from typing import Optional
from torch import autocast, GradScaler  
from utils.dataset import SeaTurtleDataset

scaler = GradScaler()

from modules.protonetloss import compute_prototypes

def train_one(
    model,
    train_dataset: SeaTurtleDataset,
    task_sampler,
    loss_fn,
    optimizer,
    device=None,
    client_id=None,
    global_prototypes: Optional[dict] = None,
    lambda_align: float = 0.5,
):
    device = device if device is not None else torch.device("cpu")

    model.to(device)
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_episodes = 0

    # Check first if its federated (uses Subset)
    # or non-federated (full Dataset)
    base_dataset = train_dataset
    if isinstance(train_dataset, torch.utils.data.Subset):
        base_dataset = train_dataset.dataset
    idx_to_identity = base_dataset.get_idx_to_identity()

    iterator = tqdm(
        task_sampler,
        desc=f"(Client {client_id}) Train" if client_id is not None else "Train",
        leave=True
    )

    for batch_indices in iterator:
        batch = [train_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()

        with autocast(device_type=str(device)):
            embeddings = model(imgs)
            embeddings = F.normalize(embeddings, p=2, dim=1)
            loss, acc = loss_fn(embeddings, labels)

            align_loss = torch.tensor(0.0, device=device)
            if global_prototypes is not None and len(global_prototypes) > 0:
                local_prototypes_tensor, local_classes = compute_prototypes(
                    embeddings, labels, loss_fn.n_support
                )
                
                local_identities = [idx_to_identity[int(c)] for c in local_classes]
                
                align_losses = []
                for i, identity in enumerate(local_identities):
                    if identity in global_prototypes:
                        local_proto = F.normalize(local_prototypes_tensor[i], dim=0)
                        global_proto = F.normalize(
                            global_prototypes[identity].to(device), dim=0
                        )
                        
                        # Cosine distance between the two prototypes per identity
                        align_loss_i = 1.0 - F.cosine_similarity(
                            local_proto.unsqueeze(0),
                            global_proto.unsqueeze(0),
                            dim=1
                        )
                        align_losses.append(align_loss_i)
                
                if len(align_losses) > 0:
                    align_loss = torch.stack(align_losses).mean()

            loss = loss + lambda_align * align_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        # Track losses
        running_loss += loss.item()
        running_acc += acc.item()
        num_episodes += 1

        iterator.set_postfix({
            'loss': f'{loss.item():.2f}',
            'acc': f'{acc.item()*100:.1f}%'
        })

    n = max(num_episodes, 1)

    loss = running_loss / n
    acc = running_acc / n

    return loss, acc


# def batch_all_triplet_loss(embeddings, labels, margin=0.3):
#     """
#     Batch-all triplet loss (Re-ID standard from "In Defense of Triplet Loss").
    
#     For each anchor:
#     - All positives contribute to loss
#     - All hard negatives (violating margin) contribute to loss
    
#     This is MORE effective than batch-hard for small batches.
#     """
#     n = embeddings.size(0)
    
#     # Pairwise distances [N, N] (using squared Euclidean for efficiency)
#     dist_mat = torch.cdist(embeddings, embeddings, p=2).pow(2)
    
#     # Create masks
#     labels_equal = labels.unsqueeze(1) == labels.unsqueeze(0)  # [N, N]
#     labels_not_equal = ~labels_equal
    
#     # Mask out self-comparisons
#     mask_self = torch.eye(n, dtype=torch.bool, device=embeddings.device)
#     labels_equal = labels_equal & ~mask_self
    
#     # Get all valid triplets
#     anchor_positive_dist = dist_mat.unsqueeze(2)  # [N, N, 1]
#     anchor_negative_dist = dist_mat.unsqueeze(1)  # [N, 1, N]
    
#     # Triplet loss for all combinations
#     triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    
#     # Mask to get valid triplets (anchor != positive, anchor != negative, positive != negative)
#     mask = labels_equal.unsqueeze(2) & labels_not_equal.unsqueeze(1)
#     mask = mask & (triplet_loss > 0)  # Only hard triplets
    
#     # Average over valid triplets
#     triplet_loss = triplet_loss * mask.float()
#     num_valid = mask.sum().float()
    
#     if num_valid > 0:
#         return triplet_loss.sum() / num_valid
#     else:
#         return torch.tensor(0.0, device=embeddings.device)


# def center_loss(embeddings, labels, centers, alpha=0.5):
#     """
#     Center loss: pulls embeddings toward their class centers.
#     Often combined with triplet loss in Re-ID.
    
#     Args:
#         embeddings: [N, D]
#         labels: [N]
#         centers: [num_classes, D] (learnable parameters)
#         alpha: learning rate for center updates
    
#     Returns:
#         loss, updated_centers
#     """
#     batch_size = embeddings.size(0)
    
#     # Get centers for current batch
#     centers_batch = centers[labels]  # [N, D]
    
#     # Compute center loss
#     loss = (embeddings - centers_batch).pow(2).sum() / 2.0 / batch_size
    
#     # Update centers (moving average)
#     with torch.no_grad():
#         for i in range(batch_size):
#             label = labels[i].item()
#             centers[label] = centers[label] * (1 - alpha) + embeddings[i] * alpha
    
#     return loss, centers