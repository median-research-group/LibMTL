import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class GLS(AbsWeighting):
    r"""Geometric Loss Strategy (GLS).
    
    This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
    and implemented by us.

    """
    def __init__(self):
        super(GLS, self).__init__()
        
    def backward(self, losses, **kwargs):
        loss = torch.pow(losses.prod(), 1./self.task_num)
        loss.backward()
        batch_weight = losses / (self.task_num * losses.prod())
        return batch_weight.detach().cpu().numpy()