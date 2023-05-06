import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

try:
    import cvxpy as cp
except ModuleNotFoundError:
    from pip._internal import main as pip
    pip(['install', '--user', 'cvxpy'])
    import cvxpy as cp

class Nash_MTL(AbsWeighting):
    r"""Nash-MTL.
    
    This method is proposed in `Multi-Task Learning as a Bargaining Game (ICML 2022) <https://proceedings.mlr.press/v162/navon22a/navon22a.pdf>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/AvivNavon/nash-mtl>`_. 

    Args:
        update_weights_every (int, default=1): Period of weights update.
        optim_niter (int, default=20): The max iteration of optimization solver.
        max_norm (float, default=1.0): The max norm of the gradients.


    .. warning::
            Nash_MTL is not supported by representation gradients, i.e., ``rep_grad`` must be ``False``.

    """
    def __init__(self):
        super(Nash_MTL, self).__init__()
        
    def init_param(self):
        self.step = 0
        self.prvs_alpha_param = None
        self.init_gtg = np.eye(self.task_num)
        self.prvs_alpha = np.ones(self.task_num, dtype=np.float32)
        self.normalization_factor = np.ones((1,))
        
    def _stop_criteria(self, gtg, alpha_t):
        return (
            (self.alpha_param.value is None)
            or (np.linalg.norm(gtg @ alpha_t - 1 / (alpha_t + 1e-10)) < 1e-3)
            or (
                np.linalg.norm(self.alpha_param.value - self.prvs_alpha_param.value)
                < 1e-6
            )
        )
        
    def solve_optimization(self, gtg: np.array):
        self.G_param.value = gtg
        self.normalization_factor_param.value = self.normalization_factor

        alpha_t = self.prvs_alpha
        for _ in range(self.optim_niter):
            self.alpha_param.value = alpha_t
            self.prvs_alpha_param.value = alpha_t

            try:
                self.prob.solve(solver=cp.ECOS, warm_start=True, max_iters=100)
            except:
                self.alpha_param.value = self.prvs_alpha_param.value

            if self._stop_criteria(gtg, alpha_t):
                break

            alpha_t = self.alpha_param.value

        if alpha_t is not None:
            self.prvs_alpha = alpha_t

        return self.prvs_alpha
        
    def _calc_phi_alpha_linearization(self):
        G_prvs_alpha = self.G_param @ self.prvs_alpha_param
        prvs_phi_tag = 1 / self.prvs_alpha_param + (1 / G_prvs_alpha) @ self.G_param
        phi_alpha = prvs_phi_tag @ (self.alpha_param - self.prvs_alpha_param)
        return phi_alpha

    def _init_optim_problem(self):
        self.alpha_param = cp.Variable(shape=(self.task_num,), nonneg=True)
        self.prvs_alpha_param = cp.Parameter(
            shape=(self.task_num,), value=self.prvs_alpha
        )
        self.G_param = cp.Parameter(
            shape=(self.task_num, self.task_num), value=self.init_gtg
        )
        self.normalization_factor_param = cp.Parameter(
            shape=(1,), value=np.array([1.0])
        )

        self.phi_alpha = self._calc_phi_alpha_linearization()

        G_alpha = self.G_param @ self.alpha_param
        constraint = []
        for i in range(self.task_num):
            constraint.append(
                -cp.log(self.alpha_param[i] * self.normalization_factor_param)
                - cp.log(G_alpha[i])
                <= 0
            )
        obj = cp.Minimize(
            cp.sum(G_alpha) + self.phi_alpha / self.normalization_factor_param
        )
        self.prob = cp.Problem(obj, constraint)
        
    def backward(self, losses, **kwargs):
        self.update_weights_every = kwargs['update_weights_every']
        self.optim_niter = kwargs['optim_niter']
        self.max_norm = kwargs['max_norm']
        
        if self.step == 0:
            self._init_optim_problem()
        if (self.step % self.update_weights_every) == 0:
            self.step += 1

            if self.rep_grad:
                raise ValueError('No support method Nash_MTL with representation gradients (rep_grad=True)')
            else:
                self._compute_grad_dim()
                grads = self._compute_grad(losses, mode='autograd')
            
            GTG = torch.mm(grads, grads.t())
            self.normalization_factor = torch.norm(GTG).detach().cpu().numpy().reshape((1,))
            GTG = GTG / self.normalization_factor.item()
            alpha = self.solve_optimization(GTG.cpu().detach().numpy())
        else:
            self.step += 1
            alpha = self.prvs_alpha
            
        alpha = torch.from_numpy(alpha).to(torch.float32).to(self.device)
        torch.sum(alpha*losses).backward()

        if self.max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.get_share_params(), self.max_norm)
        
        return alpha.detach().cpu().numpy()