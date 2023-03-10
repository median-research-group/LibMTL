import torch, math, warnings
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.loss import AbsLoss

warnings.simplefilter('ignore', UserWarning)

# pawsx
class SCLoss(AbsLoss):
    def __init__(self, label_num):
        super(SCLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss()
        self.label_num = label_num
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred.view(-1, self.label_num), gt.view(-1))
