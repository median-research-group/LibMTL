import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """
    def __init__(self, metric_name: list = ['Acc']):
        all_metric_info = {'Acc': 1}
        super(AccMetric, self).__init__(metric_name, all_metric_info)
        
    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        return {'Acc': (sum(self.record)/sum(self.bs))}
