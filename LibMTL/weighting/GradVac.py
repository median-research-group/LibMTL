import torch, random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class GradVac(AbsWeighting):
    r"""Gradient Vaccine (GradVac).
    
    This method is proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR 2021 Spotlight) <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
    and implemented by us.

    Args:
        beta (float, default=0.5): The exponential moving average (EMA) decay parameter.

    .. warning::
            GradVac is not supported with representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(GradVac, self).__init__()
        
    def backward(self, losses, **kwargs):
        beta = kwargs['beta']
        try:
            a = self.rho_T
        except:
            self.rho_T = torch.zeros(self.task_num, self.task_num).to(self.device)
        if self.rep_grad:
            raise ValueError('No support method GradVac with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward') # [task_num, grad_dim]
        batch_weight = np.ones(len(losses))
        pc_grads = grads.clone()
        for tn_i in range(self.task_num):
            task_index = list(range(self.task_num))
            task_index.remove(tn_i)
            random.shuffle(task_index)
            for tn_j in task_index:
                rho_ij = torch.dot(pc_grads[tn_i], grads[tn_j]) / (pc_grads[tn_i].norm()*grads[tn_j].norm())
                if rho_ij < self.rho_T[tn_i, tn_j]:
                    w = pc_grads[tn_i].norm()*(self.rho_T[tn_i, tn_j]*(1-rho_ij**2).sqrt()-rho_ij*(1-self.rho_T[tn_i, tn_j]**2).sqrt())/(grads[tn_j].norm()*(1-self.rho_T[tn_i, tn_j]**2).sqrt())
                    pc_grads[tn_i] += grads[tn_j]*w
                    batch_weight[tn_j] += w.item()
                    self.rho_T[tn_i, tn_j] = (1-beta)*self.rho_T[tn_i, tn_j] + beta*rho_ij
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        return batch_weight
