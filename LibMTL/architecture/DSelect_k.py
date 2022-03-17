import torch, math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.MMoE import MMoE

class DSelect_k(MMoE):
    r"""DSelect-k.
    
    This method is proposed in `DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning (NeurIPS 2021) <https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html>`_ \
    and implemented by modifying from the `official TensorFlow implementation <https://github.com/google-research/google-research/tree/master/dselect_k_moe>`_. 

    Args:
        img_size (list): The size of input data. For example, [3, 244, 244] denotes input images with size 3x224x224.
        num_experts (int): The number of experts shared by all the tasks. Each expert is an encoder network.
        num_nonzeros (int): The number of selected experts.
        kgamma (float, default=1.0): A scaling parameter for the smooth-step function.

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(DSelect_k, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        self._num_nonzeros = self.kwargs['num_nonzeros']
        self._gamma = self.kwargs['kgamma']
        
        self._num_binary = math.ceil(math.log2(self.num_experts))
        self._power_of_2 = (self.num_experts == 2 ** self._num_binary)
        
        self._z_logits = nn.ModuleDict({task: nn.Linear(self.input_size, 
                                                        self._num_nonzeros*self._num_binary) for task in self.task_name})
        self._w_logits = nn.ModuleDict({task: nn.Linear(self.input_size, self._num_nonzeros) for task in self.task_name})
        
        # initialization
        for param in self._z_logits.parameters():
            param.data.uniform_(-self._gamma/100, self._gamma/100)
        for param in self._w_logits.parameters():
            param.data.uniform_(-0.05, 0.05)
        
        binary_matrix = np.array([list(np.binary_repr(val, width=self._num_binary)) \
                                  for val in range(self.num_experts)]).astype(bool)
        self._binary_codes = torch.from_numpy(binary_matrix).to(self.device).unsqueeze(0)  
        
        self.gate_specific = None
        
    def _smooth_step_fun(self, t, gamma=1.0):
        return torch.where(t<=-gamma/2, torch.zeros_like(t, device=t.device),
                   torch.where(t>=gamma/2, torch.ones_like(t, device=t.device),
                         (-2/(gamma**3))*(t**3) + (3/(2*gamma))*t + 1/2))
    
    def _entropy_reg_loss(self, inputs):
        loss = -(inputs*torch.log(inputs+1e-6)).sum() * 1e-6
        if not self._power_of_2:
            loss += (1/inputs.sum(-1)).sum()
        loss.backward(retain_graph=True)
    
    def forward(self, inputs, task_name=None):
        experts_shared_rep = torch.stack([e(inputs) for e in self.experts_shared])
        out = {}
        for task in self.task_name:
            if task_name is not None and task != task_name:
                continue
            sample_logits = self._z_logits[task](torch.flatten(inputs, start_dim=1))
            sample_logits = sample_logits.reshape(-1, self._num_nonzeros, 1, self._num_binary)
            smooth_step_activations = self._smooth_step_fun(sample_logits)
            selector_outputs = torch.where(self._binary_codes.unsqueeze(0), smooth_step_activations, 
                                           1 - smooth_step_activations).prod(3)
            selector_weights = F.softmax(self._w_logits[task](torch.flatten(inputs, start_dim=1)), dim=1)
            expert_weights = torch.einsum('ij, ij... -> i...', selector_weights, selector_outputs)
            gate_rep = torch.einsum('ij, ji... -> i...', expert_weights, experts_shared_rep)
            gate_rep = self._prepare_rep(gate_rep, task, same_rep=False)
            out[task] = self.decoders[task](gate_rep)
        
        if self.training:
            self._entropy_reg_loss(selector_outputs)
        return out
