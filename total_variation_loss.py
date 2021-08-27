import torch
import torch.nn as nn

class TotalVariationLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y):
        # taken from https://towardsdatascience.com/pytorch-implementation-of-perceptual-losses-for-real-time-style-transfer-8d608e2e9902
        return (torch.sum(torch.abs(y[:, :, :, :-1] - y[:, :, :, 1:]))
                + torch.sum(torch.abs(y[:, :, :-1, :] - y[:, :, 1:, :])))