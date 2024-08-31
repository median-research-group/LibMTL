import torch
from LibMTL.weighting.abstract_weighting import AbsWeighting

class ExcessMTL(AbsWeighting):
    r"""ExcessMTL.
    
    This method is proposed in `Robust Multi-Task Learning with Excess Risks (ICML 2024) <https://openreview.net/forum?id=JzWFmMySpn>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/yifei-he/ExcessMTL/blob/main/LibMTL/LibMTL/weighting/ExcessMTL.py>`_. 

    """
    def __init__(self):
        super(ExcessMTL, self).__init__()
        
    def init_param(self):
        self.loss_weight = torch.tensor([1.0]*self.task_num, device=self.device, requires_grad=False)
        self.grad_sum = None
        self.first_epoch = True
        
    def backward(self, losses, **kwargs):
        
        grads = self._get_grads(losses, mode='autograd')
        if self.rep_grad:
            per_grads, grads = grads[0], grads[1]

            grads = []
            for grad in per_grads:
                grads.append(torch.sum(grad, dim=0))
            grads = torch.stack(grads)
        
        if self.grad_sum is None:
            self.grad_sum = torch.zeros_like(grads)
        
        w = torch.zeros(self.task_num, device=self.device)
        for i in range(self.task_num):
            self.grad_sum[i] += grads[i]**2
            grad_i = grads[i]
            h_i = torch.sqrt(self.grad_sum[i] + 1e-7)
            w[i] = grad_i * (1 / h_i) @ grad_i.t()

        if self.first_epoch:
            self.initial_w = w
            self.first_epoch = False
        else:
            w = w / self.initial_w
            robust_step_size = kwargs['robust_step_size']
            self.loss_weight = self.loss_weight * torch.exp(w* robust_step_size)
            self.loss_weight = self.loss_weight / self.loss_weight.sum() * self.task_num
            self.loss_weight = self.loss_weight.detach().clone()

        self.encoder.zero_grad()
        loss = torch.mul(losses, self.loss_weight).sum()
        loss.backward()

        return self.loss_weight.cpu().numpy()