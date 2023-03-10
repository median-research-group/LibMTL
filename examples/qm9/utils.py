import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss

class QM9Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self, std, scale=1):
        super(QM9Metric, self).__init__()

        self.std = std
        self.scale = scale
        
    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred * (self.std).to(pred.device) - gt * (self.std).to(pred.device)).view(pred.size()[0], -1).sum(-1)
        self.record.append(abs_err.cpu().numpy())
        
    def score_fun(self):
        r"""
        """
        records = np.concatenate(self.record)
        return [records.mean()*self.scale]
