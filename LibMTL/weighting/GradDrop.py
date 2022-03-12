import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class GradDrop(AbsWeighting):
    r"""Gradient Sign Dropout (GradDrop).
    
    This method is proposed in `Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html>`_ \
    and implemented by us.

    Args:
        leak (float, default=0.0): The leak parameter for the weighting matrix.

    .. warning::
            GradDrop is not supported by parameter gradients, i.e., ``rep_grad`` must be ``True``.

    """
    def __init__(self):
        super(GradDrop, self).__init__()
        
    def backward(self, losses, **kwargs):
        leak = kwargs['leak']
        if self.rep_grad:
            per_grads = self._compute_grad(losses, mode='backward', rep_grad=True)
        else:
            raise ValueError('No support method GradDrop with parameter gradients (rep_grad=False)')
        
        if not isinstance(self.rep, dict):
            inputs = self.rep.unsqueeze(0).repeat_interleave(self.task_num, dim=0)
        else:
            try:
                inputs = torch.stack(list(self.rep.values()))
                per_grads = torch.stack(per_grads)
            except:
                raise ValueError('The representation dimensions of different tasks must be consistent')
        grads = (per_grads*inputs.sign()).sum(1)
        P = 0.5 * (1 + grads.sum(0) / (grads.abs().sum(0)+1e-7))
        U = torch.rand_like(P)
        M = P.gt(U).unsqueeze(0).repeat_interleave(self.task_num, dim=0)*grads.gt(0) + \
            P.lt(U).unsqueeze(0).repeat_interleave(self.task_num, dim=0)*grads.lt(0)
        M = M.unsqueeze(1).repeat_interleave(per_grads.size()[1], dim=1)
        transformed_grad = (per_grads*(leak+(1-leak)*M))
        
        if not isinstance(self.rep, dict):
            self.rep.backward(transformed_grad.sum(0))
        else:
            for tn, task in enumerate(self.task_name):
                self.rep[task].backward(transformed_grad[tn], retain_graph=True)
        return None