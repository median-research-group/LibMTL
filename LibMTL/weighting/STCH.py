import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class STCH(AbsWeighting):
    r"""STCH.
    
    This method is proposed in `Smooth Tchebycheff Scalarization for Multi-Objective Optimization (ICML 2024) <https://openreview.net/forum?id=m4dO5L6eCp>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/Xi-L/STCH/tree/main/STCH_MTL>`_. 

    """
    def __init__(self):
        super(STCH, self).__init__()
        
    def init_param(self):
        self.step = 0
        self.nadir_vector = None
        
        self.average_loss = 0.0
        self.average_loss_count = 0
        
    def backward(self, losses, **kwargs):
        self.step += 1
        mu = kwargs['STCH_mu']
        warmup_epoch = kwargs['STCH_warmup_epoch']
        
        batch_weight = np.ones(len(losses))
        
        if self.epoch < warmup_epoch:
            loss = torch.mul(torch.log(losses+1e-20), torch.ones_like(losses).to(self.device)).sum()
            loss.backward()
            return  batch_weight 
        elif self.epoch == warmup_epoch:
            loss = torch.mul(torch.log(losses+1e-20), torch.ones_like(losses).to(self.device)).sum()
            self.average_loss += losses.detach() 
            self.average_loss_count += 1
            
            loss.backward()
            return  batch_weight 
        else:
            if self.nadir_vector == None:
                self.nadir_vector = self.average_loss / self.average_loss_count
                print(self.nadir_vector)
            
            losses = torch.log(losses/self.nadir_vector+1e-20)
            max_term = torch.max(losses.data).detach()
            reg_losses = losses - max_term
           
            loss = mu * torch.log(torch.sum(torch.exp(reg_losses/mu))) * self.task_num
            loss.backward()
            
            return  batch_weight 