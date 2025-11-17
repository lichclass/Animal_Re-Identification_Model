# This class gives the user the flexibility to
# use their own encoder network.

import torch.nn as nn

class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)
        result = x.flatten(start_dim=1)
        return result