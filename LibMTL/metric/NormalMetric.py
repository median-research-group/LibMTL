import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class NormalMetric(AbsMetric):
    def __init__(self, metric_name: list = ['mean', 'median', '<11.25', '<22.5', '<30']):
        all_metric_info = {'mean': 0, 'median': 0, '<11.25': 1, '<22.5': 1, '<30': 1}
        super(NormalMetric, self).__init__(metric_name, all_metric_info)
        
    def update_fun(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(pred*gt, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        self.record.append(error)

    def score_fun(self):
        records = np.concatenate(self.record)
        score =  {'mean': np.mean(records), 
                  'median': np.median(records), 
                  '<11.25': np.mean((records < 11.25)*1.0), 
                  '<22.5': np.mean((records < 22.5)*1.0), 
                  '<30': np.mean((records < 30)*1.0)}
        return {m: score[m] for m in self.metric_name}