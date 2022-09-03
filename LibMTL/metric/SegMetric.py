import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metric import AbsMetric

class SegMetric(AbsMetric):
    def __init__(self, num_classes: int, metric_name: list = ['mIoU', 'pixAcc']):
        all_metric_info = {'mIoU': 1, 'pixAcc': 1}
        super(SegMetric, self).__init__(metric_name, all_metric_info)
        
        self.num_classes = num_classes
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)
        
    def update_fun(self, pred, gt):
        self.record = self.record.to(pred.device)
        pred = pred.softmax(1).argmax(1).flatten()
        gt = gt.long().flatten()
        k = (gt >= 0) & (gt < self.num_classes)
        inds = self.num_classes * gt[k].to(torch.int64) + pred[k]
        self.record += torch.bincount(inds, minlength=self.num_classes**2).reshape(self.num_classes, self.num_classes)
        
    def score_fun(self):
        h = self.record.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        acc = torch.diag(h).sum() / h.sum()
        score = {'mIoU': torch.mean(iu).item(), 'pixAcc': acc.item()}
        return {m: score[m] for m in self.metric_name}
    
    def reinit(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)