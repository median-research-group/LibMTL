import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class L1Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self, metric_name: list = ['L1']):
        all_metric_info = {'L1': 0}
        super(L1Metric, self).__init__(metric_name, all_metric_info)
        
    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred - gt)
        self.record.append(abs_err.item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        records = np.array(self.record)
        batch_size = np.array(self.bs)
        return {'L1': (records*batch_size).sum()/(sum(batch_size))}
