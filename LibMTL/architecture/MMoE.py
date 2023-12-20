import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class MMoE(AbsArchitecture):
    r"""Multi-gate Mixture-of-Experts (MMoE).
    
    This method is proposed in `Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ \
    and implemented by us.

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (int): The number of experts shared for all tasks. Each expert is an encoder network.

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(MMoE, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self.img_size = self.kwargs['img_size']
        self.input_size = np.array(self.img_size, dtype=int).prod()
        self.num_experts = self.kwargs['num_experts'][0]
        self.experts_shared = nn.ModuleList([encoder_class() for _ in range(self.num_experts)])
        self.gate_specific = nn.ModuleDict({task: nn.Sequential(nn.Linear(self.input_size, self.num_experts),
                                                                nn.Softmax(dim=-1)) for task in self.task_name})
        
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            selector = self.gate_specific[task](torch.flatten(inputs, start_dim=1)) 
            gate_rep = torch.einsum('ij..., ji -> j...', experts_shared_rep, selector)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        return out
    
    def get_share_params(self):
        return self.experts_shared.parameters()

    def zero_grad_share_params(self):
        self.experts_shared.zero_grad(set_to_none=False)
