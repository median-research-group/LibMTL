import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.optimize import least_squares

from LibMTL.weighting.abstract_weighting import AbsWeighting

class FairGrad(AbsWeighting):
    r"""FairGrad.
    
    This method is proposed in `Fair Resource Allocation in Multi-Task Learning (ICML 2024) <https://openreview.net/forum?id=KLmWRMg6nL>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/OptMN-Lab/fairgrad>`_. 

    """
    def __init__(self):
        super().__init__()
        
    def backward(self, losses, **kwargs):
        alpha = kwargs['FairGrad_alpha']

        if self.rep_grad:
            raise ValueError('No support method FairGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='autograd')
            
        GTG = torch.mm(grads, grads.t())

        x_start = np.ones(self.task_num) / self.task_num
        A = GTG.data.cpu().numpy()

        def objfn(x):
            return np.dot(A, x) - np.power(1 / x, 1 / alpha)

        res = least_squares(objfn, x_start, bounds=(0, np.inf))
        w_cpu = res.x
        ww = torch.Tensor(w_cpu).to(self.device)

        torch.sum(ww*losses).backward()

        return w_cpu
