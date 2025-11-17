import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import models, transforms, datasets
from torchvision.models.segmentation.deeplabv3 import ASPP

from tqdm import tqdm

# ============================================
# 1. Your ResNet18 + ASPP encoder
# ============================================
class ResNet18ASPPEncoder(nn.Module):
    """
    ResNet18 backbone + ASPP from DeepLabv3 (Chen et al. 2017).

    Flow:
        Pretrained ResNet18 (up to last conv) -> ASPP -> Global Avg Pool -> Linear -> L2-normalized embedding
    """
    def __init__(self, embedding_dim: int = 256, use_pretrained: bool = True):
        super().__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
        # [B, 512, H/32, W/32]
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        self.aspp = ASPP(
            in_channels=512,
            atrous_rates=(12, 24, 36),
            out_channels=256,
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)          # [B, 512, h, w]
        x = self.aspp(x)              # [B, 256, h, w]
        x = self.pool(x)              # [B, 256, 1, 1]
        x = x.flatten(1)              # [B, 256]
        x = self.fc(x)                # [B, embedding_dim]
        x = F.normalize(x, p=2, dim=1)
        return x


# ============================================
# 2. Your ProtoNet + PrototypicalLoss + TaskSampler
#    (import from your existing files)
# ============================================
from modules.protonet import ProtoNet
from modules.protonetloss import PrototypicalLoss
from modules.tasksampler import TaskSampler  # adjust path if needed


# ============================================
# 3. Train one epoch (episodic)
# ============================================
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


# ============================================
# 4. Simple evaluation (optional but useful)
# ============================================
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


# ============================================
# 5. CIFAR100 sanity-check harness
# ============================================
def sanity_check_cifar100(
    n_way=5,
    n_support=5,
    n_query=15,
    train_episodes=100,
    val_episodes=40,
    lr=1e-4,
    weight_decay=1e-4,
):
    """
    Quick sanity check:
    - Use CIFAR100 as a fake few-shot dataset.
    - Sample N-way episodes with your TaskSampler.
    - Check if training acc > random (1/N).
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Dataset & transforms ---
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    full_train = datasets.CIFAR100(
        root="data",
        train=True,
        transform=transform,
        download=True,
    )

    # For a quick sanity check, just pretend the whole CIFAR100 train split is "meta-train"
    # If you want, you can make a small val split by taking a subset of indices later.
    train_dataset = full_train
    val_dataset = full_train  # reuse for now, it's just a sanity check

    # CIFAR100 stores labels in .targets
    all_labels = train_dataset.targets  # list of ints

    n_samples = n_support + n_query

    # --- Task samplers for train & val ---
    train_task_sampler = TaskSampler(
        labels=all_labels,
        n_way=n_way,
        n_samples=n_samples,
        iterations=train_episodes,
        allow_replacement=False,
    )

    val_task_sampler = TaskSampler(
        labels=all_labels,
        n_way=n_way,
        n_samples=n_samples,
        iterations=val_episodes,
        allow_replacement=False,
    )

    # --- Model + loss + optimizer ---
    encoder = ResNet18ASPPEncoder(embedding_dim=256, use_pretrained=True)
    model = ProtoNet(encoder)
    loss_fn = PrototypicalLoss(n_support=n_support)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- Run a few "epochs" of episodes ---
    for epoch in range(1, 4):  # e.g. 3 sanity epochs
        train_loss, train_acc = train_one_epoch(
            model=model,
            train_dataset=train_dataset,
            task_sampler=train_task_sampler,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc = evaluate_one_epoch(
            model=model,
            eval_dataset=val_dataset,
            task_sampler=val_task_sampler,
            loss_fn=loss_fn,
            device=device,
        )

        print(
            f"[Epoch {epoch}] "
            f"Train loss: {train_loss:.4f}, acc: {train_acc:.4f} | "
            f"Val loss: {val_loss:.4f}, acc: {val_acc:.4f}"
        )

    print("Sanity check on CIFAR100 completed.")


if __name__ == "__main__":
    sanity_check_cifar100()
