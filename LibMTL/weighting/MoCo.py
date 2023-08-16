import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class MoCo(AbsWeighting):
    r"""MoCo.
    
    This method is proposed in `Mitigating Gradient Bias in Multi-objective Learning: A Provably Convergent Approach (ICLR 2023) <https://openreview.net/forum?id=dLAYGdKTi2>`_ \
    and implemented based on the author' sharing code (Heshan Fernando: fernah@rpi.edu). 

    Args:
        MoCo_beta (float, default=0.5): The learning rate of y.
        MoCo_beta_sigma (float, default=0.5): The decay rate of MoCo_beta.
        MoCo_gamma (float, default=0.1): The learning rate of lambd.
        MoCo_gamma_sigma (float, default=0.5): The decay rate of MoCo_gamma.
        MoCo_rho (float, default=0): The \ell_2 regularization parameter of lambda's update.

    .. warning::
            MoCo is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(MoCo, self).__init__()

    def init_param(self):
        self._compute_grad_dim()
        self.step = 0
        self.y = torch.zeros(self.task_num, self.grad_dim).to(self.device)
        self.lambd = (torch.ones([self.task_num, ]) / self.task_num).to(self.device)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        beta, beta_sigma = kwargs['MoCo_beta'], kwargs['MoCo_beta_sigma']
        gamma, gamma_sigma = kwargs['MoCo_gamma'], kwargs['MoCo_gamma_sigma']
        rho = kwargs['MoCo_rho']

        if self.rep_grad:
            raise ValueError('No support method MoCo with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')

        with torch.no_grad():
            for tn in range(self.task_num):
                grads[tn] = grads[tn]/(grads[tn].norm()+1e-8)*losses[tn]
        self.y = self.y - (beta/self.step**beta_sigma) * (self.y - grads)
        self.lambd = F.softmax(self.lambd - (gamma/self.step**gamma_sigma) * (self.y@self.y.t()+rho*torch.eye(self.task_num).to(self.device))@self.lambd, -1)
        new_grads = self.y.t()@self.lambd

        self._reset_grad(new_grads)
        return self.lambd.detach().cpu().numpy()
