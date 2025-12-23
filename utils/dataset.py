import os
import ast
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import pandas as pd

class SeaTurtleDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None, train=True, verbose=False, identity_to_idx=None):
        self.df = dataframe.reset_index(drop=True)
        self.root_dir = root_dir
        self.train = train
        self.verbose = verbose
        
        if transform is None:
            if train:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.RandAugment(num_ops=2, magnitude=9),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])
            else:
                self.transform = T.Compose([
                    T.Resize((224, 224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transform = transform

        # Create label mappings
        if identity_to_idx is None:
            self.identity_to_idx = {ident: i for i, ident in enumerate(sorted(self.df["identity"].unique()))}
        else:
            self.identity_to_idx = identity_to_idx
        self.idx_to_identity = {v: k for k, v in self.identity_to_idx.items()}
        
        if verbose:
            print(f"\n=== SeaTurtleDataset Loaded ===")
            print(f"Root Dir      : {self.root_dir}")
            print(f"Total Samples : {len(self.df)}")
            print(f"Unique IDs    : {len(self.identity_to_idx)}")
            print(f"Mode          : {'TRAIN' if train else 'EVAL'}")
            print(f"Cropping Mode : BOUNDING BOX\n")
    
    def get_idx_to_identity(self):
        return self.idx_to_identity
    
    def get_identity_to_idx(self):
        return self.identity_to_idx
    
    def get_num_classes(self):  # ← Added: useful for ArcFace initialization
        return len(self.identity_to_idx)
    
    def __len__(self):
        return len(self.df)
    
    def _load_and_crop_image(self, row):
        img_path = os.path.join(self.root_dir, row["file_name"])
        
        if not os.path.exists(img_path):
            if self.verbose:
                print(f"[WARN] Missing image: {img_path}")
            img = Image.new("RGB", (224, 224), color=(0, 0, 0))
            return img
        
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
            
            # Add padding to bounding box
            padding = 0.1 
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
        img = self._load_and_crop_image(row)
        img = self.transform(img)
        
        identity = row["identity"]

        # sanitize identity (important)
        if pd.isna(identity):
            raise ValueError(f"NaN identity at idx={idx}, file={row.get('file_name')}")

        identity = str(identity).strip()

        if self.train:
            # crash early if mapping mismatch (better than CUDA crash later)
            label = self.identity_to_idx[identity]
        else:
            label = self.identity_to_idx.get(identity, -1)

        return img, int(label), identity
