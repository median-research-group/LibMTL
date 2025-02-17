import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class FAMO(AbsWeighting):
    r"""Fast Adaptive Multitask Optimization (FAMO).
    
    This method is proposed in `FAMO: Fast Adaptive Multitask Optimization (NeurIPS 2023) <https://openreview.net/forum?id=zMeemcUeXL>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/FAMO>`_. 

    Args:
        FAMO_w_lr (float, default=0.025): The learing rate of loss weights.
        FAMO_w_gamma (float, default=1e-3): The weight decay of loss weights.

    """
    def __init__(self):
        super().__init__()
    
    def init_param(self):
        self.step = 0
        self.min_losses = torch.zeros(self.task_num).to(self.device)
        self.w = torch.tensor([0.0] * self.task_num, device=self.device, requires_grad=True)
        self.w_opt = torch.optim.Adam([self.w], lr=0.0, weight_decay=0.0)
        
    def backward(self, losses, **kwargs):
        self.step += 1
        if self.step == 1:
            for param_group in self.w_opt.param_groups:
                param_group['lr'] = kwargs['FAMO_w_lr']
                param_group['weight_decay'] = kwargs['FAMO_w_gamma']

        self.prev_losses = losses
        z = F.softmax(self.w, -1)
        D = losses - self.min_losses + 1e-8
        c = (z / D).sum().detach()
        loss = (D.log() * z / c).sum()
        loss.backward()
        return None

    def update_w(self, curr_losses):
        delta = (self.prev_losses - self.min_losses + 1e-8).log() - \
                (curr_losses      - self.min_losses + 1e-8).log()
        with torch.enable_grad():
            d = torch.autograd.grad(F.softmax(self.w, -1),
                                    self.w,
                                    grad_outputs=delta.detach())[0]
        self.w_opt.zero_grad(set_to_none=False)
        self.w.grad = d
        self.w_opt.step()
