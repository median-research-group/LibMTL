import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.architecture.abstract_arch import AbsArchitecture

class _transform_resnet_cross(nn.Module):
    def __init__(self, resnet_network, task_name, device):
        super(_transform_resnet_cross, self).__init__()
        
        self.task_name = task_name
        self.task_num = len(task_name)
        self.device = device
#         self.forward_task = None
        
        self.shared_conv = nn.Sequential(resnet_network.conv1, resnet_network.bn1, 
                                         resnet_network.relu, resnet_network.maxpool)
        self.shared_layer = nn.ModuleDict({})
        for i in range(4):
            self.shared_layer[str(i)] = nn.ModuleList([eval('resnet_network.layer'+str(i+1))]*self.task_num)
        self.cross_unit = nn.Parameter(torch.ones(4, self.task_num))
        
    def forward(self, inputs):
        s_rep = self.shared_conv(inputs)
        ss_rep = {i: [0]*self.task_num for i in range(4)}
        for i in range(4):
            for tn in range(self.task_num):
                if i == 0:
                    ss_rep[i][tn] = self.shared_layer[str(i)][tn](s_rep)
                else:
                    cross_rep = sum([self.cross_unit[i-1][j]*ss_rep[i-1][j] for j in range(self.task_num)])
                    ss_rep[i][tn] = self.shared_layer[str(i)][tn](cross_rep)
        return ss_rep[3]

class Cross_stitch(AbsArchitecture):
    r"""Cross-stitch Networks (Cross_stitch).
    
    This method is proposed in `Cross-stitch Networks for Multi-task Learning (CVPR 2016) <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>`_ \
    and implemented by us. 

    .. warning::
            - :class:`Cross_stitch` does not work with multiple inputs MTL problem, i.e., ``multi_input`` must be ``False``.

            - :class:`Cross_stitch` is only supported with ResNet-based encoder.

    """
    def __init__(self, task_name, encoder, decoders, rep_grad, multi_input, device, **kwargs):
        super(Cross_stitch, self).__init__(task_name, encoder, decoders, rep_grad, multi_input, device, **kwargs)
        
        if self.multi_input:
            raise ValueError('No support Cross Stitch for multiple inputs MTL problem')
            
        try: 
            callable(eval('encoder.layer1'))
            self.encoder = _transform_resnet_cross(encoder.to(device), task_name, device)
        except:
            self.encoder.resnet_network = _transform_resnet_cross(self.encoder.resnet_network.to(device), task_name, device)
        