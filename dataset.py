import numpy as np
import ast

from PIL import Image
from torch.utils.data import Dataset

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
        encounter_id = int(self.df.loc[idx, "encounter_label"])
        if isinstance(bbox, float) and np.isnan(bbox):
            bbox = None
        if isinstance(bbox, str):
            bbox = ast.literal_eval(bbox)
        if bbox is not None and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
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

        return img, label, encounter_id