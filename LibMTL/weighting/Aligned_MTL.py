import torch, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class Aligned_MTL(AbsWeighting):
    r"""Aligned-MTL.
    
    This method is proposed in `Independent Component Alignment for Multi-Task Learning (CVPR 2023) <https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/SamsungLabs/MTL>`_. 

    """
    def __init__(self):
        super(Aligned_MTL, self).__init__()
        
    def backward(self, losses, **kwargs):

        grads = self._get_grads(losses, mode='backward')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]
        
        M = torch.matmul(grads, grads.t()) # [num_tasks, num_tasks]
        lmbda, V = torch.symeig(M, eigenvectors=True)
        tol = (
            torch.max(lmbda)
            * max(M.shape[-2:])
            * torch.finfo().eps
        )
        rank = sum(lmbda > tol)

        order = torch.argsort(lmbda, dim=-1, descending=True)
        lmbda, V = lmbda[order][:rank], V[:, order][:, :rank]

        sigma = torch.diag(1 / lmbda.sqrt())
        B = lmbda[-1].sqrt() * ((V @ sigma) @ V.t())
        alpha = B.sum(0)

        if self.rep_grad:
            self._backward_new_grads(alpha, per_grads=per_grads)
        else:
            self._backward_new_grads(alpha, grads=grads)
        return alpha.detach().cpu().numpy()
