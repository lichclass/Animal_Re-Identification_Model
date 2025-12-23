from torch.utils.data import Dataset
from PIL import Image
import ast
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.models import convnext_base, ConvNeXt_Base_Weights
import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from pathlib import Path
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE
from torch.optim.lr_scheduler import CosineAnnealingLR

# =========================
# ### NEW (PK SAMPLER): imports
# =========================
from torch.utils.data import Sampler
import random
from collections import defaultdict


class SeaTurtleDataset(Dataset):
    def __init__(self, df, split_mode, split, transform=None):
        self.df = df[df[f'split_{split_mode}'] == split].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.loc[idx, "file_name"]
        label = int(self.df.loc[idx, "label"])

        img = Image.open(img_path).convert("RGB")

        bbox = self.df.loc[idx, "bounding_box"]

        # =========================
        # ### CHANGED: handle NaN bbox safely
        # =========================
        if isinstance(bbox, float) and np.isnan(bbox):
            bbox = None

        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)

        # =========================
        # ### CHANGED: clamp bbox to image bounds to avoid invalid crops
        # =========================
        if bbox is not None:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)

            W, H = img.size
            x1 = max(0, min(x1, W - 1))
            y1 = max(0, min(y1, H - 1))
            x2 = max(1, min(x2, W))
            y2 = max(1, min(y2, H))

            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))

        if self.transform:
            img = self.transform(img)

        return img, label


# =========================
# ### NEW (PK SAMPLER): identity-balanced P×K sampler
# =========================
# class PKBatchSampler(Sampler):
#     def __init__(self, dataset, P=8, K=4, drop_last=True, seed=42):
#         self.dataset = dataset
#         self.P = int(P)
#         self.K = int(K)
#         self.drop_last = drop_last
#         self.seed = int(seed)
#         self.epoch = 0  # ### FIX

#         label_to_indices = defaultdict(list)
#         for idx in range(len(dataset)):
#             lab = int(dataset.df.loc[idx, "label"])
#             label_to_indices[lab].append(idx)

#         self.label_to_indices = dict(label_to_indices)

#         # ### FIX: keep only labels with >= K samples
#         self.labels = [lab for lab, idxs in self.label_to_indices.items() if len(idxs) >= self.K]
#         if len(self.labels) == 0:
#             raise ValueError(f"No labels have >= K={self.K} samples. Lower K.")

#         self.num_samples = len(dataset)
#         self.batch_size = self.P * self.K
#         self.num_batches = self.num_samples // self.batch_size if self.drop_last else int(math.ceil(self.num_samples / self.batch_size))

#     def set_epoch(self, epoch: int):  # ### FIX
#         self.epoch = int(epoch)

#     def __len__(self):
#         return self.num_batches

#     def __iter__(self):
#         rng = random.Random(self.seed + self.epoch)  # ### FIX: change per epoch

#         pools = {}
#         for lab in self.labels:
#             idxs = self.label_to_indices[lab].copy()
#             rng.shuffle(idxs)
#             pools[lab] = idxs

#         ptr = {lab: 0 for lab in pools.keys()}

#         for _ in range(self.num_batches):
#             chosen_labels = rng.sample(self.labels, self.P) if len(self.labels) >= self.P else [rng.choice(self.labels) for _ in range(self.P)]

#             batch = []
#             for lab in chosen_labels:
#                 idxs = pools[lab]
#                 start = ptr[lab]
#                 end = start + self.K

#                 if end > len(idxs):
#                     rng.shuffle(idxs)
#                     start = 0
#                     end = self.K
#                     ptr[lab] = 0

#                 # idxs length is guaranteed >= K now
#                 batch.extend(idxs[start:end])
#                 ptr[lab] += self.K

#             if len(batch) == self.batch_size:
#                 yield batch
#             elif not self.drop_last and len(batch) > 0:
#                 yield batch



class ConvNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True, dropout=0.1):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)

        in_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        self.backbone = model

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, embedding_dim)

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.dropout(feat)
        emb = self.proj(feat)
        return emb


class ArcFace(nn.Module):
    def __init__(self, embedding_size, classnum, s=64.0, m=0.5):
        super().__init__()
        self.classnum = classnum
        self.kernel = nn.Parameter(torch.Tensor(embedding_size, classnum))
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m
        self.s = s
        self.eps = 1e-4

    def forward(self, embeddings, label):
        embeddings = F.normalize(embeddings, dim=1)
        kernel_norm = F.normalize(self.kernel, dim=0)

        cosine = torch.mm(embeddings, kernel_norm)
        cosine = cosine.clamp(-1 + self.eps, 1 - self.eps)

        if label is None:
            return cosine * self.s, embeddings

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.view(-1, 1), self.m)

        theta = torch.acos(cosine)
        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = torch.cos(theta_m)

        return cosine_m * self.s, embeddings


class ConvNeXtArcFaceModel(nn.Module):
    def __init__(self, num_classes, embedding_dim=512, dropout=0.1):
        super().__init__()
        self.backbone = ConvNeXtBackbone(
            embedding_dim=embedding_dim,
            pretrained=True,
            dropout=dropout
        )
        self.head = ArcFace(
            embedding_size=embedding_dim,
            classnum=num_classes,
            s=64.0,
            m=0.5
        )

    def forward(self, x, labels=None):
        emb = self.backbone(x)
        logits, emb_norm = self.head(emb, labels)
        return logits, emb_norm


@torch.no_grad()
def extract_embeddings(model, loader, device, max_points=None):
    all_embs = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extract embeddings"):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        _, emb = model(images, labels=None)
        all_embs.append(emb.detach().cpu())
        all_labels.append(labels.detach().cpu())

    embs = torch.cat(all_embs, dim=0).float()
    labs = torch.cat(all_labels, dim=0).long()

    if max_points is not None:
        embs = embs[:max_points]
        labs = labs[:max_points]

    return embs, labs


@torch.no_grad()
def compute_rank1_rank5_map(query_emb, query_lab, gallery_emb, gallery_lab):
    sim = query_emb @ gallery_emb.t()
    sorted_idx = torch.argsort(sim, dim=1, descending=True)
    sorted_gallery_labels = gallery_lab[sorted_idx]

    rank1 = 0
    rank5 = 0
    ap_sum = 0.0
    valid_q = 0

    for i in range(sim.size(0)):
        q = query_lab[i]
        rel = (sorted_gallery_labels[i] == q).float()
        num_rel = int(rel.sum().item())
        if num_rel == 0:
            continue
        valid_q += 1

        if rel[0].item() == 1.0:
            rank1 += 1
        if rel[:5].sum().item() > 0:
            rank5 += 1

        cumsum_rel = torch.cumsum(rel, dim=0)
        idx = torch.arange(1, rel.numel() + 1, device=rel.device).float()
        precision_at_k = cumsum_rel / idx
        ap = (precision_at_k * rel).sum().item() / num_rel
        ap_sum += ap

    if valid_q == 0:
        return 0.0, 0.0, 0.0

    return rank1 / valid_q, rank5 / valid_q, ap_sum / valid_q


def main():
    split_mode = 'closed'
    segment = 'head'

    DATASET_DIR = "./data/turtle-data"
    METADATA_FILE = Path(DATASET_DIR) / f'metadata_splits_{segment}.csv'
    metadata_df = pd.read_csv(METADATA_FILE)

    metadata_df['file_name'] = metadata_df['file_name'].apply(lambda x: str(Path(DATASET_DIR) / x))
    metadata_df = metadata_df[['file_name', 'identity', 'date', f'split_{split_mode}', 'bounding_box']]

    identities = metadata_df["identity"].unique().tolist()
    label_to_index = {label: idx for idx, label in enumerate(identities)}
    idx_to_identity = {label_to_index[label]: label for label in identities}
    metadata_df['label'] = metadata_df['identity'].map(label_to_index)

    metadata_df["encounter_id"] = metadata_df["identity"].astype(str) + "_" + metadata_df["date"].astype(str)

    transforms_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_test = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_set = SeaTurtleDataset(metadata_df, split_mode, 'train', transform=transforms_train)
    train_gallery_set = SeaTurtleDataset(metadata_df, split_mode, 'train', transform=transforms_test)
    val_set = SeaTurtleDataset(metadata_df, split_mode, 'valid', transform=transforms_test)
    test_set = SeaTurtleDataset(metadata_df, split_mode, 'test', transform=transforms_test)

    train_ids = set(train_set.df["label"].tolist())
    val_ids = set(val_set.df["label"].tolist())
    test_ids = set(test_set.df["label"].tolist())
    print("VAL missing-from-train:", len(val_ids - train_ids))
    print("TEST missing-from-train:", len(test_ids - train_ids))

    print("Train/Val/Test sizes:", len(train_set), len(val_set), len(test_set))

    epochs = 50
    learning_rate = 1e-2
    model_path = 'best_model.pth'

    model = ConvNeXtArcFaceModel(num_classes=len(identities), embedding_dim=512, dropout=0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model.to(device)

    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded existing model weights.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    results_dir = Path("/results/")
    results_name_dir = results_dir / f"convnext_arcface_{split_mode}_{segment}_PK"
    results_name_dir.mkdir(parents=True, exist_ok=True)

    tsne_plots_dir = results_name_dir / 'tsne_plots'
    tsne_plots_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True

    # =========================
    # ### NEW (PK SAMPLER): configure P and K so P*K == batch size
    # =========================
    P = 8
    K = 4
    assert P * K == BATCH_SIZE, "Set P and K so that P*K == BATCH_SIZE"

    # =========================
    # ### NEW (PK SAMPLER): replace shuffle=True with batch_sampler=PKBatchSampler(...)
    # IMPORTANT: when batch_sampler is used, DO NOT set batch_size/shuffle in DataLoader.
    # =========================
    # pk_batch_sampler = PKBatchSampler(train_set, P=P, K=K, drop_last=True, seed=42)

    train_loader = DataLoader(
        train_set,
        # batch_sampler=pk_batch_sampler,
        batch_size=BATCH_SIZE,
        shuffle=True,                             
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # unchanged loaders
    train_gallery_loader = DataLoader(train_gallery_set, batch_size=BATCH_SIZE, shuffle=False,
                                      num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    early_stopping_patience = 5
    best_rank1 = 0.0
    best_rank5 = 0.0
    best_map = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "val_rank1": [], "val_rank5": [], "val_map": []}

    eval_every = 1

    for epoch in range(epochs):
        # pk_batch_sampler.set_epoch(epoch)  # ### FIX: change sampler order per epoch

        model.train()
        running_loss = 0.0
        seen = 0  # ### FIX

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            seen += images.size(0)
        # scheduler.step()
        epoch_loss = running_loss / max(1, seen)  # ### FIX
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f} (seen={seen})")

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_set)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        do_eval = (epoch == 0) or ((epoch + 1) % eval_every == 0)

        if do_eval:
            gallery_emb, gallery_lab = extract_embeddings(model, train_gallery_loader, device)
            query_emb, query_lab = extract_embeddings(model, val_loader, device)

            gallery_emb = F.normalize(gallery_emb, dim=1)
            query_emb = F.normalize(query_emb, dim=1)

            rank1, rank5, mAP = compute_rank1_rank5_map(query_emb, query_lab, gallery_emb, gallery_lab)

            print(f"Validation Re-ID Results (Gallery=train(no-aug), Query=val):")
            print(f"  Rank-1: {rank1*100:.2f}%")
            print(f"  Rank-5: {rank5*100:.2f}%")
            print(f"  mAP:    {mAP*100:.2f}%")

            history["val_rank1"].append(rank1)
            history["val_rank5"].append(rank5)
            history["val_map"].append(mAP)

            if rank1 > best_rank1:
                best_rank1 = rank1
                best_rank5 = rank5
                best_map = mAP
                patience_counter = 0
                torch.save(model.state_dict(), model_path)
                print(f"✅ Model saved! Best Rank-1 now: {best_rank1*100:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered (no Rank-1 improvement).")
                    break

            emb, lab = extract_embeddings(model, test_loader, device)
            emb = emb.cpu().numpy()
            lab = lab.cpu().numpy()

            max_points = 2000
            if len(emb) > max_points:
                idx = np.random.RandomState(42).choice(len(emb), size=max_points, replace=False)
                emb = emb[idx]
                lab = lab[idx]

            emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

            tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
            emb_2d = tsne.fit_transform(emb)

            plt.figure(figsize=(10, 8))
            plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=8, c=lab, alpha=0.7)
            plt.title("t-SNE of ConvNeXt + ArcFace Embeddings (Test subset)")
            plt.xlabel("t-SNE 1")
            plt.ylabel("t-SNE 2")
            plt.colorbar(label="Class label (index)")
            plt.savefig(tsne_plots_dir / f"tsne_plot_epoch_{epoch+1}.png")

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(results_name_dir / "loss_curve.png")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    gallery_emb, gallery_lab = extract_embeddings(model, train_gallery_loader, device)
    query_emb, query_lab = extract_embeddings(model, test_loader, device)

    gallery_emb = F.normalize(gallery_emb, dim=1)
    query_emb = F.normalize(query_emb, dim=1)

    rank1, rank5, mAP = compute_rank1_rank5_map(query_emb, query_lab, gallery_emb, gallery_lab)
    print(f"Re-ID Results (Gallery=train(no-aug), Query=test):")
    print(f"  Rank-1: {rank1*100:.2f}%")
    print(f"  Rank-5: {rank5*100:.2f}%")
    print(f"  mAP:    {mAP*100:.2f}%")

    with open(results_name_dir / "test_results.txt", "w") as f:
        f.write(f"Re-ID Results (Gallery=train(no-aug), Query=test):\n")
        f.write(f"  Rank-1: {rank1*100:.2f}%\n")
        f.write(f"  Rank-5: {rank5*100:.2f}%\n")
        f.write(f"  mAP:    {mAP*100:.2f}%\n")


if __name__ == "__main__":
    main()
