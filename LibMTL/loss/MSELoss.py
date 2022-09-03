import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss

class MSELoss(AbsLoss):
    r"""The Mean Squared Error (MSE) loss function.
    """
    def __init__(self):
        super(MSELoss, self).__init__()
        
        self.loss_fn = nn.MSELoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss