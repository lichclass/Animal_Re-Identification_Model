import os
import ast
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import pandas as pd

class SeaTurtleDataset(Dataset):
    """
    Dataset for SeaTurtleID2022 metadata_splits_* CSVs.

    Each row contains:
        file_name
        identity
        bounding_box (string "[x, y, w, h]" or None)
        category
        ... other metadata

    This version CROPS using the bounding box.
    """

    def __init__(self, dataframe, root_dir, transform=None, verbose=False):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.verbose = verbose

        # Default transforms for ProtoNet / ResNet
        if transform is None:
            self.transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
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
            print(f"ID Mapping    : {self.identity_to_idx}")
            print("Cropping Mode : BOUNDING BOX\n")

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

            x = max(0, int(x))
            y = max(0, int(y))
            w = min(int(w), img_width - x)
            h = min(int(h), img_height - y)

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
