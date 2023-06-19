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
        GradVac_beta (float, default=0.5): The exponential moving average (EMA) decay parameter.
        GradVac_group_type (int, default=0): The parameter granularity (0: whole_model; 1: all_layer; 2: all_matrix).

    .. warning::
            GradVac is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(GradVac, self).__init__()
        
    def init_param(self):
        self.step = 0

    def _init_rho(self, group_type):
        if group_type == 0: # whole_model
            self.k_idx = [-1]
        elif group_type == 1: # all_layer
            self.k_idx = []
            for module in self.encoder.modules():
                if len(module._modules.items()) == 0 and len(module._parameters) > 0:
                    self.k_idx.append(sum([w.data.numel() for w in module.parameters()]))
        elif group_type == 2: # all_matrix
            self._compute_grad_dim()
            self.k_idx = self.grad_index
        else:
            raise ValueError
        self.rho_T = torch.zeros(self.task_num, self.task_num, len(self.k_idx)).to(self.device)

        
    def backward(self, losses, **kwargs):
        beta = kwargs['GradVac_beta']
        group_type = kwargs['GradVac_group_type']
        if self.step == 0:
            self._init_rho(group_type)

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
                for k in range(len(self.k_idx)):
                    beg, end = sum(self.k_idx[:k]), sum(self.k_idx[:k+1])
                    if end == -1:
                        end = grads.size()[-1]
                    rho_ijk = torch.dot(pc_grads[tn_i,beg:end], grads[tn_j,beg:end]) / (pc_grads[tn_i,beg:end].norm()*grads[tn_j,beg:end].norm()+1e-8)
                    if rho_ijk < self.rho_T[tn_i, tn_j, k]:
                        w = pc_grads[tn_i,beg:end].norm()*(self.rho_T[tn_i,tn_j,k]*(1-rho_ijk**2).sqrt()-rho_ijk*(1-self.rho_T[tn_i,tn_j,k]**2).sqrt())/(grads[tn_j,beg:end].norm()*(1-self.rho_T[tn_i,tn_j,k]**2).sqrt()+1e-8)
                        pc_grads[tn_i,beg:end] += grads[tn_j,beg:end]*w
                        # batch_weight[tn_j] += w.item()
                        self.rho_T[tn_i,tn_j,k] = (1-beta)*self.rho_T[tn_i,tn_j,k] + beta*rho_ijk
        new_grads = pc_grads.sum(0)
        self._reset_grad(new_grads)
        self.step += 1
        return batch_weight
