import os
import ast
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import pandas as pd

class SeaTurtleDataset(Dataset):
    """
    Enhanced SeaTurtleDataset with better augmentation for Re-ID.
    """

    def __init__(self, dataframe, root_dir, transform=None, train=True, verbose=False):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.train = train
        self.verbose = verbose

        # IMPROVED TRANSFORMS for Re-ID
        if transform is None:
            if train:
                # Training: aggressive augmentation
                self.transform = T.Compose([
                    T.Resize((256, 256)),  # Larger initial size
                    T.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Random crop
                    T.RandomHorizontalFlip(p=0.5),
                    
                    # Color augmentation (critical for turtle shells)
                    T.ColorJitter(
                        brightness=0.3,  # Increased from 0.2
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.15
                    ),
                    
                    # Random perspective (handles viewpoint changes)
                    T.RandomPerspective(distortion_scale=0.2, p=0.3),
                    
                    # Random rotation (turtles can be at any angle)
                    T.RandomRotation(degrees=15),
                    
                    # Random erasing (occlusion simulation)
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    ),
                    T.RandomErasing(p=0.3, scale=(0.02, 0.15)),
                ])
            else:
                # Validation/Test: minimal augmentation
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]
                    )
                ])
        else:
            self.transform = transform

        # Identity → integer label mapping
        self.identity_to_idx = {
            ident: i for i, ident in enumerate(sorted(self.df["identity"].unique()))
        }
        self.idx_to_identity = {v: k for k, v in self.identity_to_idx.items()}

        if verbose:
            print(f"\n=== SeaTurtleDataset Loaded ===")
            print(f"Root Dir      : {self.root_dir}")
            print(f"Total Samples : {len(self.df)}")
            print(f"Unique IDs    : {len(self.identity_to_idx)}")
            print(f"Mode          : {'TRAIN' if train else 'EVAL'}")
            print(f"Cropping Mode : BOUNDING BOX\n")

    def __len__(self):
        return len(self.df)

    def _load_and_crop_image(self, row):
        img_path = os.path.join(self.root_dir, row["file_name"])

        # Default → black image if missing
        if not os.path.exists(img_path):
            if self.verbose:
                print(f"[WARN] Missing image: {img_path}")
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            return img

        # Load image
        img = Image.open(img_path).convert("RGB")
        img_width, img_height = img.size

        # Check bounding box
        bbox_str = row.get("bounding_box", None)

        if bbox_str is None or pd.isna(bbox_str) or bbox_str == "None":
            # No bbox → use whole image
            if self.verbose:
                print(f"[INFO] No bbox for {row['file_name']} → using full image")
            return img

        # Parse bbox: "[x, y, w, h]"
        try:
            bbox = ast.literal_eval(bbox_str)
            x, y, w, h = bbox

            # Add padding to bounding box (important for Re-ID!)
            padding = 0.1  # 10% padding
            x = max(0, int(x - w * padding))
            y = max(0, int(y - h * padding))
            w = min(int(w * (1 + 2*padding)), img_width - x)
            h = min(int(h * (1 + 2*padding)), img_height - y)

            cropped = img.crop((x, y, x + w, y + h))
            return cropped

        except Exception as e:
            if self.verbose:
                print(f"[ERR] Bad bbox for {row['file_name']}: {bbox_str} → {e}")
                print("[INFO] Using full image instead.")
            return img

    def __getitem__(self, idx):
        row = self.df.loc[idx]

        # Load & crop
        img = self._load_and_crop_image(row)
        img = self.transform(img)

        # Identity → integer class
        identity = row["identity"]
        label = self.identity_to_idx[identity]

        return img, label


# ========================================
# USAGE IN experiment.py
# ========================================
# Update preprocess_data() function:
#
# train_dataset = SeaTurtleDataset(train_df, args.data_dir, train=True, verbose=False)
# val_dataset   = SeaTurtleDataset(val_df, args.data_dir, train=False, verbose=False)
# test_dataset  = SeaTurtleDataset(test_df, args.data_dir, train=False, verbose=False)