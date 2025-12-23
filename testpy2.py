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
        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)
        if bbox is not None:
            x, y, w, h = bbox
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            img = img.crop((x1, y1, x2, y2))
        if self.transform:
            img = self.transform(img)
        return img, label


class ConvNeXtBackbone(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True, dropout=0.1):
        super().__init__()
        weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_base(weights=weights)
        
        in_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        self.backbone = model
        
        # Projection with dropout
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
        cosine = cosine.clamp(-1+self.eps, 1-self.eps)

        if label is None:
            return cosine * self.s, embeddings

        m_hot = torch.zeros(label.size()[0], cosine.size()[1], device=cosine.device)
        m_hot.scatter_(1, label.view(-1, 1), self.m)

        theta = torch.acos(cosine)

        theta_m = torch.clip(theta + m_hot, min=self.eps, max=math.pi - self.eps)
        cosine_m = torch.cos(theta_m)
        scaled_cosine_m = cosine_m * self.s
        return scaled_cosine_m, embeddings

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
        emb = self.backbone(x)               # (B, D) raw
        logits, emb_norm = self.head(emb, labels)  # works for labels=None too
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
    sim = query_emb @ gallery_emb.t()  # (Q, G)
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

    # metadata configs
    split_mode = 'closed'
    segment = 'head'

    DATASET_DIR = './data/turtle-data'
    METADATA_FILE = Path(DATASET_DIR) / f'metadata_splits_{segment}.csv'
    metadata_df = pd.read_csv(METADATA_FILE)
    metadata_df['file_name'] = metadata_df['file_name'].apply(lambda x: str(Path(DATASET_DIR) / x))
    metadata_df = metadata_df[['file_name', 'identity', f'split_{split_mode}', 'bounding_box']]
    metadata_df = metadata_df.rename(columns={'file': 'file_name', 'id': 'identity'})
    identities = metadata_df["identity"].unique().tolist()
    label_to_index = {label: idx for idx, label in enumerate(identities)}
    identities = {label_to_index[label]: label for label in identities}
    metadata_df['label'] = metadata_df['identity'].map(label_to_index)
    print(metadata_df.head(-10))
        
    transforms_train = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    transforms_test  = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        
    train_set = SeaTurtleDataset(metadata_df, split_mode, 'train', transform=transforms_train)
    train_gallery_set = SeaTurtleDataset(metadata_df, split_mode, 'train', transform=transforms_test)
    val_set = SeaTurtleDataset(metadata_df, split_mode, 'valid', transform=transforms_test)
    test_set = SeaTurtleDataset(metadata_df, split_mode, 'test', transform=transforms_test)
    print("Train/Val/Test sizes:", len(train_set), len(val_set), len(test_set))

    epochs = 50
    learning_rate = 0.001

    model_path = 'best_model.pth'
    model = ConvNeXtArcFaceModel(num_classes=len(identities), embedding_dim=512, dropout=0.1)
    if Path(model_path).exists():
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model weights.")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

    results_dir = Path("./results")
    results_name_dir = results_dir / f"convnext_arcface_{split_mode}_{segment}_2"
    results_name_dir.mkdir(parents=True, exist_ok=True)

    BATCH_SIZE = 32
    NUM_WORKERS = 4
    PIN_MEMORY = True   
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    train_gallery_loader = DataLoader(train_gallery_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.to(device)

    early_stopping_patience = 5
    best_rank1 = 0.0
    patience_counter = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {epoch_loss:.4f}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                images, labels = images.to(device), labels.to(device)
                logits, _ = model(images, labels)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Validation Loss: {val_loss:.4f}")

        history["train_loss"].append(epoch_loss)
        history["val_loss"].append(val_loss)

        gallery_emb, gallery_lab = extract_embeddings(model, train_gallery_loader, device)
        query_emb, query_lab     = extract_embeddings(model, val_loader, device)

        gallery_emb = torch.nn.functional.normalize(gallery_emb, dim=1)
        query_emb   = torch.nn.functional.normalize(query_emb, dim=1)

        rank1, rank5, mAP = compute_rank1_rank5_map(query_emb, query_lab, gallery_emb, gallery_lab)
        print(f"Validation Re-ID Results (Gallery=train, Query=val):")
        print(f"  Rank-1: {rank1*100:.2f}%")
        print(f"  Rank-5: {rank5*100:.2f}%")
        print(f"  mAP:    {mAP*100:.2f}%")

        if rank1 > best_rank1:
            best_rank1 = rank1
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
            print("Model saved!")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print("Early stopping triggered.")
                break

    # Save Plot of training/validation loss
    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig(results_name_dir / "loss_curve.png")
    
    # Testing
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    gallery_emb, gallery_lab = extract_embeddings(model, train_gallery_loader, device)
    query_emb, query_lab     = extract_embeddings(model, test_loader, device)

    gallery_emb = torch.nn.functional.normalize(gallery_emb, dim=1)
    query_emb   = torch.nn.functional.normalize(query_emb, dim=1)

    rank1, rank5, mAP = compute_rank1_rank5_map(query_emb, query_lab, gallery_emb, gallery_lab)
    print(f"Re-ID Results (Gallery=train, Query=test):")
    print(f"  Rank-1: {rank1*100:.2f}%")
    print(f"  Rank-5: {rank5*100:.2f}%")
    print(f"  mAP:    {mAP*100:.2f}%")

    with open(results_name_dir / "test_results.txt", "w") as f:
        f.write(f"Re-ID Results (Gallery=train, Query=test):\n")
        f.write(f"  Rank-1: {rank1*100:.2f}%\n")
        f.write(f"  Rank-5: {rank5*100:.2f}%\n")
        f.write(f"  mAP:    {mAP*100:.2f}%\n")

    emb, lab = extract_embeddings(model, test_loader, device, max_points=2000)
    print("Embeddings:", emb.shape, "Labels:", lab.shape)
    if torch.is_tensor(emb):
        emb = emb.cpu().numpy()
    if torch.is_tensor(lab):
        lab = lab.cpu().numpy()
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    emb_2d = tsne.fit_transform(emb)

    plt.figure(figsize=(10, 8))
    plt.scatter(emb_2d[:, 0], emb_2d[:, 1], s=8, c=lab, alpha=0.7)
    plt.title("t-SNE of ConvNeXt + ArcFace Embeddings (Test subset)")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.colorbar(label="Class label (index)")
    plt.savefig(results_name_dir / "tsne_plot.png")

if __name__ == "__main__":
    main()