import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AbsMetric(object):
    r"""An abstract class for the performance metrics of a task. 

    Attributes:
        record (list): A list of the metric scores in every iteration.
        bs (list): A list of the number of data in every iteration.
    """
    def __init__(self):
        self.record = []
        self.bs = []
    
    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass
    
    @property
    def score_fun(self):
        r"""Calculate the final score (when a epoch ends).

        Return:
            list: A list of metric scores.
        """
        pass
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when a epoch ends).
        """
        self.record = []
        self.bs = []
    
# accuracy
class AccMetric(AbsMetric):
    r"""Calculate the accuracy.
    """
    def __init__(self):
        super(AccMetric, self).__init__()
        
    def update_fun(self, pred, gt):
        r"""
        """
        pred = F.softmax(pred, dim=-1).max(-1)[1]
        self.record.append(gt.eq(pred).sum().item())
        self.bs.append(pred.size()[0])
        
    def score_fun(self):
        r"""
        """
        return [(sum(self.record)/sum(self.bs))]
