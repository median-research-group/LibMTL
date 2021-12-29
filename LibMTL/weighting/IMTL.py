import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class IMTL(AbsWeighting):
    r"""Impartial Multi-task Learning (IMTL).
    
    This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021) <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(IMTL, self).__init__()
    
    def init_param(self):
        self.loss_scale = nn.Parameter(torch.tensor([0.0]*self.task_num, device=self.device))
        
    def backward(self, losses, **kwargs):
        losses = self.loss_scale.exp()*losses - self.loss_scale
        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        
        grads_unit = grads/torch.norm(grads, p=2, dim=-1, keepdim=True)

        D = grads[0:1].repeat(self.task_num-1, 1) - grads[1:]
        U = grads_unit[0:1].repeat(self.task_num-1, 1) - grads_unit[1:]

        alpha = torch.matmul(torch.matmul(grads[0], U.t()), torch.inverse(torch.matmul(D, U.t())))
        alpha = torch.cat((1-alpha.sum().unsqueeze(0), alpha), dim=0)
        
        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()
