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
    def __init__(self, metric_name: list, all_metric_info: dict):
        self.record = []
        self.bs = []
        self.metric_name = metric_name
        self.metric_info = {k: v for k, v in all_metric_info.items() if k in metric_name}
    
    @property
    def update_fun(self, pred, gt):
        r"""Calculate the metric scores in every iteration and update :attr:`record`.

        Args:
            pred (torch.Tensor): The prediction tensor.
            gt (torch.Tensor): The ground-truth tensor.
        """
        pass
    
    @property
    def score_fun(self) -> dict:
        r"""Calculate the final score (when an epoch ends).

        Return:
            list: A list of metric scores.
        """
        return {m: 0 for m in self.metric_name}
    
    def reinit(self):
        r"""Reset :attr:`record` and :attr:`bs` (when an epoch ends).
        """
        self.record = []
        self.bs = []
