import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).
    
    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(RLW, self).__init__()
        
    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()
