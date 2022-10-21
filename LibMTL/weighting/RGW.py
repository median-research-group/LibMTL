import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class RGW(AbsWeighting):
    r"""Random Gradient Weighting (RGW).
    
    This method is proposed in `Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-Task Learning (TMLR 2022) <https://openreview.net/forum?id=jjtFD8A1Wx>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(RGW, self).__init__()
        
    def backward(self, losses, **kwargs):
        batch_weight = F.softmax(torch.randn(self.task_num), dim=-1).to(self.device)
        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        if self.rep_grad:
            self._backward_new_grads(batch_weight, per_grads=per_grads)
        else:
            self._backward_new_grads(batch_weight, grads=grads)
        return batch_weight.detach().cpu().numpy()
