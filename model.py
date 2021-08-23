import torch
import torch.nn as nn

class ByLayerModel(nn.Module):
    def __init__(self, sequence, names=None):
        super().__init__()
        self.sequence = sequence
        self.names = names


    def forward(self, x):
        res = []
        for layer in self.sequence:
            x = layer(x)
            res.append(x)

        if self.names is None:
            return res
        else:
            return {name: r for name, r in zip(self.names, res)}
    