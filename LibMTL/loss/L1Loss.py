import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss

class L1Loss(AbsLoss):
    r"""The Mean Absolute Error (MAE) loss function.
    """
    def __init__(self):
        super(L1Loss, self).__init__()
        
        self.loss_fn = nn.L1Loss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss