import torch.nn as nn
import numpy as np
import torch
EPSILON_FP16 = 1e-5
from funcs import vison_utils
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class custom_L1Loss(nn.Module):
    def __init__(self,loss_func=nn.L1Loss()):

        super().__init__()
        self.func = loss_func
        self.sigmoid = nn.Sigmoid()

    def forward(self, pred, actual):
        #pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        mask=actual[1]
        loss= self.func(pred[mask].squeeze(),torch.tensor(actual[0][mask].squeeze(),dtype=torch.float32))
        return loss