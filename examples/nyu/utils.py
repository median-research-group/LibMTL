import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss

# seg
class SegMetric(AbsMetric):
    def __init__(self):
        super(SegMetric, self).__init__()
        
        self.num_classes = 13
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
        return [torch.mean(iu).item(), acc.item()]
    
    def reinit(self):
        self.record = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

# depth
class DepthMetric(AbsMetric):
    def __init__(self):
        super(DepthMetric, self).__init__()
        
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
        return [(records[i]*batch_size).sum()/(sum(batch_size)) for i in range(2)]
    
    def reinit(self):
        self.abs_record = []
        self.rel_record = []
        self.bs = []
    
# normal
class NormalMetric(AbsMetric):
    def __init__(self):
        super(NormalMetric, self).__init__()
        
    def update_fun(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0)
        error = torch.acos(torch.clamp(torch.sum(pred*gt, 1).masked_select(binary_mask), -1, 1)).detach().cpu().numpy()
        error = np.degrees(error)
        self.record.append(error)

    def score_fun(self):
        records = np.concatenate(self.record)
        return [np.mean(records), np.median(records), \
               np.mean((records < 11.25)*1.0), np.mean((records < 22.5)*1.0), \
               np.mean((records < 30)*1.0)]
    
    
class SegLoss(AbsLoss):
    def __init__(self):
        super(SegLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-1)
        
    def compute_loss(self, pred, gt):
        return self.loss_fn(pred, gt.long())
    
class DepthLoss(AbsLoss):
    def __init__(self):
        super(DepthLoss, self).__init__()
        
    def compute_loss(self, pred, gt):
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = torch.sum(torch.abs(pred - gt) * binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss
    
class NormalLoss(AbsLoss):
    def __init__(self):
        super(NormalLoss, self).__init__()
        
    def compute_loss(self, pred, gt):
        # gt has been normalized on the NYUv2 dataset
        pred = pred / torch.norm(pred, p=2, dim=1, keepdim=True)
        binary_mask = (torch.sum(gt, dim=1) != 0).float().unsqueeze(1).to(pred.device)
        loss = 1 - torch.sum((pred*gt)*binary_mask) / torch.nonzero(binary_mask, as_tuple=False).size(0)
        return loss
