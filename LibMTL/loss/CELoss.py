import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss 

class CELoss(AbsLoss):
    r"""The cross-entropy loss function.
    """
    def __init__(self):
        super(CELoss, self).__init__()
        
        self.loss_fn = nn.CrossEntropyLoss()
        
    def compute_loss(self, pred, gt):
        r"""
        """
        loss = self.loss_fn(pred, gt)
        return loss