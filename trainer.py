# ------------------------------------------------------------
# File Name: trainer.py
# Status: 🔧 READY FOR TESTING
# Revised: November 26, 2025
# Revised by: Nash Adam Muñoz
# File Description: 
#    This file contains the code for training the model.
# ------------------------------------------------------------

import torch
from tqdm import tqdm
from modules.protonetloss import PrototypicalLoss

def train_one_epoch(model, train_dataset, task_sampler, loss_fn, optimizer, device="cuda"):
    model.to(device)
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    num_episodes = 0

    for batch_indices in tqdm(task_sampler, desc="Train episodes", leave=False):

        # ---------------------------------------------------
        # Build episodic batch
        # ---------------------------------------------------
        batch = [train_dataset[i] for i in batch_indices]
        imgs, labels = zip(*batch)

        imgs = torch.stack(imgs).to(device)
        labels = torch.tensor(labels, dtype=torch.long, device=device)

        # ---------------------------------------------------
        # Forward + Backward
        # ---------------------------------------------------
        optimizer.zero_grad()

        embeddings = model(imgs)                # [N, D]
        loss, acc = loss_fn(embeddings, labels)

        loss.backward()
        optimizer.step()

        # ---------------------------------------------------
        # Accumulate metrics
        # ---------------------------------------------------
        running_loss += loss.item()
        running_acc += acc.item()
        num_episodes += 1

    # ---------------------------------------------------
    # Episode-wise average loss and accuracy
    # ---------------------------------------------------
    avg_loss = running_loss / max(num_episodes, 1)
    avg_acc = running_acc / max(num_episodes, 1)

    return avg_loss, avg_acc