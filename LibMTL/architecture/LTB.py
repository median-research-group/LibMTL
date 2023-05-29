import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture


class _transform_resnet_ltb(nn.Module):
    def __init__(self, encoder_list, task_name, device):
        super(_transform_resnet_ltb, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        # self.epochs = epochs
        self.resnet_conv = nn.ModuleDict({task: nn.Sequential(encoder_list[tn].conv1, encoder_list[tn].bn1, 
                                                              encoder_list[tn].relu, encoder_list[tn].maxpool) for tn, task in enumerate(self.task_name)})
        self.resnet_layer = nn.ModuleDict({})
        for i in range(4):
            self.resnet_layer[str(i)] = nn.ModuleList([])
            for tn in range(self.task_num):
                encoder = encoder_list[tn]
                self.resnet_layer[str(i)].append(eval('encoder.layer'+str(i+1)))
        self.alpha = nn.Parameter(torch.ones(6, self.task_num, self.task_num))
        
    def forward(self, inputs, epoch, epochs):
        if epoch < epochs/100: # warmup
            alpha = torch.ones(6, self.task_num, self.task_num).to(self.device)
        else:
            tau = epochs/20 / np.sqrt(epoch+1) # tau decay
            alpha = F.gumbel_softmax(self.alpha, dim=-1, tau=tau, hard=True)

        ss_rep = {i: [0]*self.task_num for i in range(5)}
        for i in range(5): # i: layer idx
            for tn, task in enumerate(self.task_name): # tn: task idx
                if i == 0:
                    ss_rep[i][tn] = self.resnet_conv[task](inputs)
                else:
                    child_rep = sum([alpha[i,tn,j]*ss_rep[i-1][j] for j in range(self.task_num)]) # j: module idx
                    ss_rep[i][tn] = self.resnet_layer[str(i-1)][tn](child_rep)
        return ss_rep[4]

class LTB(AbsArchitecture):
    r"""Learning To Branch (LTB).

    This method is proposed in `Learning to Branch for Multi-Task Learning (ICML 2020) <http://proceedings.mlr.press/v119/guo20e.html>`_ \
    and implemented by us. 

    .. warning::
            - :class:`LTB` does not work with multi-input problems, i.e., ``multi_input`` must be ``False``.
            - :class:`LTB` is only supported by ResNet-based encoders.
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(LTB, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

        if self.multi_input:
            raise ValueError('No support LTB for multiple inputs MTL problem')
            
        self.encoder = nn.ModuleList([self.encoder_class() for _ in range(self.task_num)])
        self.encoder = _transform_resnet_ltb(self.encoder, task_name, device)

    def forward(self, inputs, task_name=None):
        r"""
        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs, self.epoch, self.epochs)
        same_rep = True if not isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out