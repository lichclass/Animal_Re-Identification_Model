import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm

# Training Loop
def train_one_epoch(
    model: nn.Module,
    train_dataset,
    task_sampler,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: str = "cuda",
):
    """
    Episodic training loop for one 'epoch'.
    Each iteration of task_sampler = 1 ProtoNet episode.
    """
    model.to(device)
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_episodes = 0

    for batch_indices in tqdm(task_sampler, desc="Train episodes", leave=False):
        # Build episode batch
        batch = [train_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)                         # tuples of length N

        imgs = torch.stack(imgs).to(device)                # [N, C, H, W]
        labels = torch.tensor(labels, dtype=torch.long, device=device)  # [N]

        optimizer.zero_grad()
        embeddings = model(imgs)                           # [N, D]

        loss, acc = loss_fn(embeddings, labels)            # PrototypicalLoss

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += acc.item() if hasattr(acc, "item") else float(acc)
        num_episodes += 1

    avg_loss = running_loss / max(num_episodes, 1)
    avg_acc = running_acc / max(num_episodes, 1)

    return avg_loss, avg_acc


# Evaluation Loop
@torch.no_grad()
def evaluate_one_epoch(
    model: nn.Module,
    eval_dataset,
    task_sampler,
    loss_fn: nn.Module,
    device: str = "cuda",
):
    model.to(device)
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    num_episodes = 0

    for batch_indices in tqdm(task_sampler, desc="Val episodes", leave=False):
        batch = [eval_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        embeddings = model(imgs)
        loss, acc = loss_fn(embeddings, labels)

        running_loss += loss.item()
        running_acc += acc.item() if hasattr(acc, "item") else float(acc)
        num_episodes += 1

    avg_loss = running_loss / max(num_episodes, 1)
    avg_acc = running_acc / max(num_episodes, 1)

    return avg_loss, avg_acc