import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class SDMGrad(AbsWeighting):
    r"""Stochastic Direction-oriented Multi-objective Gradient descent (SDMGrad).
    
    This method is proposed in `Direction-oriented Multi-objective Learning: Simple and Provable Stochastic Algorithms (NeurIPS 2023) <https://openreview.net/forum?id=4Ks8RPcXd9>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/OptMN-Lab/sdmgrad>`_. 

    Args:
        SDMGrad_lamda (float, default=0.3): The regularization hyperparameter.
        SDMGrad_niter (int, default=20): The update iteration of loss weights.

    """
    def __init__(self):
        super().__init__()
    
    def init_param(self):
        self.w = 1/self.task_num*torch.ones(self.task_num).to(self.device)

    def euclidean_proj_simplex(self, v, s=1):
        assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
        v = v.astype(np.float64)
        n, = v.shape  
        if v.sum() == s and np.alltrue(v >= 0):
            return v
        u = np.sort(v)[::-1]
        cssv = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - s))[0][-1]
        theta = float(cssv[rho] - s) / (rho + 1)
        w = (v - theta).clip(min=0)
        return w
        
    def backward(self, losses, **kwargs):
        # losses: [3, num_tasks] in SDMGrad
        assert self.rep_grad == False, "No support method SDMGrad with representation gradients (rep_grad=True)"

        SDMGrad_lamda, SDMGrad_niter = kwargs['SDMGrad_lamda'], kwargs['SDMGrad_niter']

        grads = []
        for i in range(3):
            grads.append(self._get_grads(losses[i], mode='backward'))

        zeta_grads, xi_grads1, xi_grads2 = grads
        GG = torch.mm(xi_grads1, xi_grads2.t())
        GG_diag = torch.diag(GG)
        GG_diag = torch.where(GG_diag < 0, torch.zeros_like(GG_diag), GG_diag)
        scale = torch.mean(torch.sqrt(GG_diag))
        GG = GG / (scale.pow(2) + 1e-8)
        Gg = torch.mean(GG, dim=1)

        self.w.requires_grad = True
        optimizer = torch.optim.SGD([self.w], lr=5, momentum=0.5)
        for i in range(SDMGrad_niter):
            optimizer.zero_grad()
            self.w.grad = torch.mv(GG, self.w.detach()) + SDMGrad_lamda * Gg
            optimizer.step()
            proj = self.euclidean_proj_simplex(self.w.data.cpu().numpy())
            self.w.data.copy_(torch.from_numpy(proj).data)
        self.w.requires_grad = False

        g0 = torch.mean(zeta_grads, dim=0)
        gw = (zeta_grads * self.w.view(-1, 1)).sum(0)
        g = (gw + SDMGrad_lamda * g0) / (1 + SDMGrad_lamda)

        self._reset_grad(g)
        return None