import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class MoDo(AbsWeighting):
    r"""Multi-objective gradient with Double sampling (MoDo).
    
    This method is proposed in `Three-Way Trade-Off in Multi-Objective Learning: Optimization, Generalization and Conflict-Avoidance (NeurIPS 2023; JMLR 2024) <https://openreview.net/forum?id=yPkbdJxQ0o>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/heshandevaka/Trade-Off-MOL>`_. 

    Args:
        MoDo_gamma (float, default=0.001): The learning rate of lambd.
        MoDo_rho (float, default=0.1): The \ell_2 regularization parameter of lambda's update.

    """
    def __init__(self):
        super().__init__()
    
    def init_param(self):
        self.lambd = 1/self.task_num*torch.ones([self.task_num, ]).to(self.device)

    def _projection2simplex(self, y):
        m = len(y)
        sorted_y = torch.sort(y, descending=True)[0]
        tmpsum = 0.0
        tmax_f = (torch.sum(y) - 1.0)/m
        for i in range(m-1):
            tmpsum += sorted_y[i]
            tmax = (tmpsum - 1)/ (i+1.0)
            if tmax > sorted_y[i+1]:
                tmax_f = tmax
                break
        return torch.max(y - tmax_f, torch.zeros(m).to(y.device))
        
    def backward(self, losses, **kwargs):
        # losses: [3, num_tasks] in MoDo
        assert self.rep_grad == False, "No support method MoDo with representation gradients (rep_grad=True)"

        MoDo_gamma, MoDo_rho = kwargs['MoDo_gamma'], kwargs['MoDo_rho']

        grads = []
        for i in range(3):
            grads.append(self._get_grads(losses[i], mode='backward'))
        grads = torch.stack(grads)

        # average the gradient in decorders if only 3rd gradient is used to update shared part
        for task in list(self.decoders.keys()):
            for p_idx, p in enumerate(self.decoders[task].parameters()):
                p.grad.data = p.grad.data/3

        self.lambd = self._projection2simplex(self.lambd - MoDo_gamma*(grads[0]@(torch.transpose(grads[1], 0, 1)@self.lambd )+MoDo_rho*self.lambd))

        self._backward_new_grads(self.lambd, grads=grads[2])
        return self.lambd.detach().cpu().numpy()
