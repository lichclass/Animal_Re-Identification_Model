import numpy as np
import ast

from PIL import Image
from torch.utils.data import Dataset

class SeaTurtleDataset(Dataset):
    def __init__(self, df, split_mode, split, transform=None):
        subset = df[df[f'split_{split_mode}'] == split].copy()
        self.file_names = subset["file_name"].values

        self.train_labels = np.full(len(subset), -1, dtype=int)
        if "train_label" in subset.columns: # Some splits may not have train labels like eval and test
            self.train_labels = subset["train_label"].fillna(-1).values.astype(int)
        self.eval_labels = subset["eval_label"].values.astype(int)
        self.encounter_labels = subset["encounter_label"].values.astype(int)

        self.bboxes = [None] * len(self.file_names)
        if "bounding_box" in subset.columns: # The full image metadata doesn't have bounding boxes
            temp_bboxes = []
            for b in subset["bounding_box"]:
                if isinstance(b, str) and b.strip():
                    try:
                        temp_bboxes.append(ast.literal_eval(b))
                    except:
                        temp_bboxes.append(None)
                elif isinstance(b, (list, tuple)):
                    temp_bboxes.append(b)
                else:
                    temp_bboxes.append(None)
            self.bboxes = temp_bboxes
        self.transform = transform

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = self.file_names[idx]
        img = Image.open(img_path).convert("RGB")
        bbox = self.bboxes[idx]

        if bbox is not None and len(bbox) == 4:
            x, y, w, h = bbox
            W, H = img.size
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2, y2 = min(W, int(x + w)), min(H, int(y + h))
            if x2 > x1 and y2 > y1:
                img = img.crop((x1, y1, x2, y2))
        
        if self.transform:
            img = self.transform(img)

        return img, self.train_labels[idx], self.eval_labels[idx], self.encounter_labels[idx]