import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class DepthMetric(AbsMetric):
    def __init__(self, metric_name: list = ['abs_err', 'rel_err']):
        all_metric_info = {'abs_err': 0, 'rel_err': 0}
        super(DepthMetric, self).__init__(metric_name, all_metric_info)
        
        self.abs_record = []
        self.rel_record = []
        
    def update_fun(self, pred, gt):
        device = pred.device
        binary_mask = (torch.sum(gt, dim=1) != 0).unsqueeze(1).to(device)
        pred = pred.masked_select(binary_mask)
        gt = gt.masked_select(binary_mask)
        abs_err = torch.abs(pred - gt)
        rel_err = torch.abs(pred - gt) / gt
        abs_err = (torch.sum(abs_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        rel_err = (torch.sum(rel_err) / torch.nonzero(binary_mask, as_tuple=False).size(0)).item()
        self.abs_record.append(abs_err)
        self.rel_record.append(rel_err)
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        records = np.stack([np.array(self.abs_record), np.array(self.rel_record)])
        batch_size = np.array(self.bs)
        score = {'abs_err': (records[0]*batch_size).sum()/(sum(batch_size)),
                 'rel_err': (records[1]*batch_size).sum()/(sum(batch_size))}
        return {m: score[m] for m in self.metric_name}
    
    def reinit(self):
        self.abs_record = []
        self.rel_record = []
        self.bs = []
