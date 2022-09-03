import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss.abstract_loss import AbsLoss

class NormalLoss(AbsLoss):
    def __init__(self):
        super(NormalLoss, self).__init__()
        
    def compute_loss(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = 1 - torch.sum((pred*gt)*binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss