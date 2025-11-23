import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm
import os
import urllib.request
import zipfile

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


# Function to download dataset
def download_dataset():
    data_dir = "data"
    dataset_dir = "turtle-data"
    
    # GitHub download URL for raw zip
    dataset_link = "https://github.com/lichclass/Animal_Re-Identification_Model/raw/main/downloads/turtle-data.zip"
    
    zip_path = os.path.join(data_dir, f"{dataset_dir}.zip")

    # Ensure root data directory exists
    if not os.path.exists(data_dir):
        print(f"Creating Data Directory: {data_dir}")
        os.makedirs(data_dir, exist_ok=True)
    else:
        print(f'Data Directory "{data_dir}" already exists. Skipping creation...')

    # Skip if dataset already extracted
    dataset_path = os.path.join(data_dir, dataset_dir)
    if os.path.exists(dataset_path):
        print(f'Dataset Directory "{dataset_dir}" already exists. Skipping download...')
        return

    # Download zip file safely
    print(f"Downloading Dataset: {dataset_dir} from {dataset_link}")
    try:
        urllib.request.urlretrieve(dataset_link, zip_path)
        print("Download complete.")
    except Exception as e:
        print("Download FAILED:", e)
        return

    # Extract zip
    print("Extracting dataset...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction complete.")
    except Exception as e:
        print("Extraction FAILED:", e)
        return
    finally:
        # Clean up: remove zip file
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("Cleaned up temporary zip file.")

    print("Dataset is ready!")


# Run Training and Evaluation
def run_training():
    