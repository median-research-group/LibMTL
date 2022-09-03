import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss

class KLDivLoss(AbsLoss):
    r"""The Kullback-Leibler divergence loss function.
    """
    def __init__(self):
        super(KLDivLoss, self).__init__()
        
        self.loss_fn = nn.KLDivLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss