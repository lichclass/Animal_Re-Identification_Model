import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import os

# --- PyTorch/Torchvision Imports ---
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.amp import autocast_mode, grad_scaler

import torchvision.transforms as T
from torchvision.models import (
    convnext_base,
    ConvNeXt_Base_Weights,

    # Methods to compare
    swin_b,
    Swin_B_Weights,
    resnet50,
    ResNet50_Weights,
    densenet121,
    DenseNet121_Weights,
)

# --- Library Imports ---
from wildlife_datasets.datasets import SeaTurtleID2022
from wildlife_tools.data import ImageDataset
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.inference import KnnClassifier
from wildlife_datasets.splits import ClosedSetSplit


os.environ['KAGGLE_USERNAME'] = "nashadammuoz"
os.environ['KAGGLE_KEY'] = "KGAT_9f227e36a409b0debe5ee7a27090bd72"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

# weight aggregation
# Expected Input Example: 
# weights_list = {
#    'client_1': {'weights': model.state_dict(), 'num_samples': 1000},
#    'client_2': {'weights': model.state_dict(), 'num_samples': 1500},
#    ...
# }

# Formula:
# W_{t+1} = sum((n_k / n) * W_k) for k in clients

# Step 1: Compute the total number of samples
total_samples = sum(client['num_samples'] for client in weights_list.values())

# Step 2: Initialize an empty state_dict for the aggregated weights
agg_weights = model.state_dict()

# Step 3: Aggregate the weights
for client in weights_list.values():
    client_weights = client['weights']
    client_samples = client['num_samples']

    weight_factor = client_samples / total
    for key in agg_weights.keys():
        agg_weights[key] += client_weights[key] * weight_factor

# Step 4: Update the global model with the aggregated weights
global_model.load_state_dict(agg_weights)
