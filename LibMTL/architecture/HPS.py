import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture


class HPS(AbsArchitecture):
    r"""Hard Parameter Sharing (HPS).

    This method is proposed in `Multitask Learning: A Knowledge-Based Source of Inductive Bias (ICML 1993) <https://dl.acm.org/doi/10.5555/3091529.3091535>`_ \
    and implemented by us. 
    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(HPS, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
