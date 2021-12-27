import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AbsArchitecture(nn.Module):
    r"""An abstract class for MTL architectures.

    Args:
        task_name (list): A list of strings for all tasks.
        encoder (torch.nn.Module): A neural network module.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, ``False`` otherwise. 
        device (torch.device): The device where model and data will be allocated. 
        kwargs (dict): A dictionary of hyperparameters of architecture methods.
     
    """
    def __init__(self, task_name, encoder, decoders, rep_grad, multi_input, device, **kwargs):
        super(AbsArchitecture, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.encoder = encoder
        self.decoders = decoders
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.device = device
        self.kwargs = kwargs
        
        if self.rep_grad:
            self.rep_tasks = {}
#             self.rep = None
#             if self.multi_input:
            self.rep = {}
    
    def forward(self, inputs, task_name=None):
        r"""The forward function of an architecture.

        Args: 
            inputs (torch.Tensor): The input data.
            task_name (str, default=None): The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
        
        Returns:
            dict: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
        """
        out = {}
        s_rep = self.encoder(inputs)
        same_rep = True if isinstance(s_rep, list) and not self.multi_input else False
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            ss_rep = s_rep[tn] if isinstance(s_rep, list) else s_rep
            ss_rep = self._prepare_rep(ss_rep, task, same_rep)
            out[task] = self.decoders[task](ss_rep)
        return out
    
    def get_share_params(self):
        r"""Return the shared parameters of the model.
        """
        return self.encoder.parameters()

    def zero_grad_share_params(self):
        r"""Set gradients of the shared parameters to zero.
        """
        self.encoder.zero_grad()
        
    def _prepare_rep(self, rep, task, same_rep=None):
        r"""A useful function to allow to compute the gradients of the representations.

        Args:
            rep (torch.Tensor): The representation of one task.
            task (str): The task name corresponding to the ``rep``.
            same_rep (bool): Is ``True`` if all tasks share the same representation.

        Return:
            torch.Tensor: a representation with the same value with the input representation while \
                          can be computed gradients if ``rep_grad`` is ``True``. 
        """
        if self.rep_grad:
            if not same_rep:
                self.rep[task] = rep
            else:
                self.rep = rep
            self.rep_tasks[task] = rep.detach().clone()
            self.rep_tasks[task].requires_grad = True
            return self.rep_tasks[task]
        else:
            return rep