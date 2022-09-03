import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss

class DepthLoss(AbsLoss):
    def __init__(self):
        super(DepthLoss, self).__init__()
        
    def compute_loss(self, pred, gt):
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss