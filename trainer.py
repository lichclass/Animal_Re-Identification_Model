import torch
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional

# Import the CORRECT prototype function
from modules.protonetloss import PrototypicalLoss, compute_prototypes, compute_alignment_loss

def train_one_epoch(
    model,
    train_dataset,
    task_sampler,
    loss_fn,
    optimizer,
    tqdm_position=0,
    client_id=None,
    device="cuda",
    global_prototypes: Optional[torch.Tensor] = None,        # [n_global_classes, D]
    global_prototype_classes: Optional[torch.Tensor] = None, # [n_global_classes] (int labels)
    lambda_align: float = 0.5,
):
    """
    Episodic training step for Federated Prototypical Networks.

    global_prototypes/global_prototype_classes are expected to be:
        - global_prototypes: tensor [n_global_classes, D]
        - global_prototype_classes: tensor [n_global_classes] with *the same label space*
          as 'labels' in this client (i.e., dataset.identity_to_idx mapping).
    """

    model.to(device)
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_episodes = 0

    iterator = tqdm(
        task_sampler,
        desc=f"(Client {client_id}) Train" if client_id is not None else "Train",
        position=tqdm_position,
        leave=True
    )

    for batch_indices in iterator:

        batch = [train_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        optimizer.zero_grad()
        embeddings = model(imgs)

        # 1) Standard prototypical loss
        proto_loss, acc = loss_fn(embeddings, labels)
        total_loss = proto_loss

        # 2) Alignment loss if global prototypes provided
        if global_prototypes is not None and global_prototype_classes is not None:
            # Compute local prototypes for this episode
            local_prototypes, local_classes = compute_prototypes(
                embeddings, labels, loss_fn.n_support
            )

            # Ensure global_prototypes on same device as local_prototypes
            g_protos = global_prototypes.to(local_prototypes.device)
            g_classes = global_prototype_classes.to(local_prototypes.device)

            align_loss = compute_alignment_loss(
                local_prototypes=local_prototypes,
                local_classes=local_classes,
                global_prototypes=g_protos,
                global_classes=g_classes,
            )

            total_loss = proto_loss + lambda_align * align_loss

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        running_acc += acc.item()
        num_episodes += 1

    avg_loss = running_loss / max(num_episodes, 1)
    avg_acc  = running_acc  / max(num_episodes, 1)

    return avg_loss, avg_acc
