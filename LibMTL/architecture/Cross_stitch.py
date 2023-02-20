import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class _transform_resnet_cross(nn.Module):
    def __init__(self, encoder_list, task_name, device):
        super(_transform_resnet_cross, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
        self.resnet_conv = nn.ModuleDict({task: nn.Sequential(encoder_list[tn].conv1, encoder_list[tn].bn1, 
                                                              encoder_list[tn].relu, encoder_list[tn].maxpool) for tn, task in enumerate(self.task_name)})
        self.resnet_layer = nn.ModuleDict({})
        for i in range(4):
            self.resnet_layer[str(i)] = nn.ModuleList([])
            for tn in range(self.task_num):
                encoder = encoder_list[tn]
                self.resnet_layer[str(i)].append(eval('encoder.layer'+str(i+1)))
        self.cross_unit = nn.Parameter(torch.ones(4, self.task_num, self.task_num))
        
    def forward(self, inputs):
        s_rep = {task: self.resnet_conv[task](inputs) for task in self.task_name}
        ss_rep = {i: [0]*self.task_num for i in range(4)}
        for i in range(4):
            for tn, task in enumerate(self.task_name):
                if i == 0:
                    ss_rep[i][tn] = self.resnet_layer[str(i)][tn](s_rep[task])
                else:
                    cross_rep = sum([self.cross_unit[i-1][tn][j]*ss_rep[i-1][j] for j in range(self.task_num)])
                    ss_rep[i][tn] = self.resnet_layer[str(i)][tn](cross_rep)
        return ss_rep[3]

class Cross_stitch(AbsArchitecture):
    r"""Cross-stitch Networks (Cross_stitch).
    
    This method is proposed in `Cross-stitch Networks for Multi-task Learning (CVPR 2016) <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>`_ \
    and implemented by us. 

    .. warning::
            - :class:`Cross_stitch` does not work with multiple inputs MTL problem, i.e., ``multi_input`` must be ``False``.

            - :class:`Cross_stitch` is only supported by ResNet-based encoders.

    """
    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(Cross_stitch, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        
        if self.multi_input:
            raise ValueError('No support Cross Stitch for multiple inputs MTL problem')
        
        self.encoder = nn.ModuleList([self.encoder_class() for _ in range(self.task_num)])
        self.encoder = _transform_resnet_cross(self.encoder, task_name, device)
