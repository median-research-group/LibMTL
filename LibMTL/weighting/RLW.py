import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting


class RLW(AbsWeighting):
    r"""Random Loss Weighting (RLW).
    
    This method is proposed in `A Closer Look at Loss Weighting in Multi-Task Learning (arXiv:2111.10603) <https://arxiv.org/abs/2111.10603>`_ \
    and implemented by us.

    Args:
        dist ({'Uniform', 'Normal', 'Dirichlet', 'Bernoulli', 'constrained_Bernoulli'}, default='Normal'): The type of distribution where the loss weigghts are sampled from.

    """
    def __init__(self):
        super(RLW, self).__init__()
        
    def backward(self, losses, **kwargs):
        dist = kwargs['dist']
        try:
            if dist in ['Uniform', 'Normal', 'Dirichlet']:
                weight = self.distribution.sample()
            elif dist == 'Bernoulli':
                while True:
                    weight = self.distribution.sample()
                    if weight.sum() != 0:
                        break
            elif dist == 'constrained_Bernoulli':
                w = random.sample(range(self.task_num), k=1)
                batch_weight = torch.zeros(self.task_num).to(self.device)
                batch_weight[w] = 1.0
            else:
                raise ValueError('No support method RLW with {}'.format(dist))
        except:
            if dist == 'Uniform':
                from torch.distributions.uniform import Uniform
#                 self.distribution = Uniform(kwargs['low'], kwargs['high'])
                self.distribution = Uniform(torch.zeros(self.task_num), torch.ones(self.task_num))
                weight = self.distribution.sample()
            elif dist == 'Normal':
                from torch.distributions.normal import Normal
#                 self.distribution = Normal(kwargs['mean'], kwargs['var'])
                self.distribution = Normal(torch.zeros(self.task_num), torch.ones(self.task_num))
                weight = self.distribution.sample()
            elif dist == 'Dirichlet':
                from torch.distributions.dirichlet import Dirichlet
#                 self.distribution = Dirichlet(kwargs['concentration'])
                self.distribution = Dirichlet(torch.ones(self.task_num))
                weight = self.distribution.sample()
            elif dist == 'Bernoulli':
                from torch.distributions.bernoulli import Bernoulli
                self.distribution = Bernoulli(torch.ones(self.task_num)*0.5)
                while True:
                    weight = self.distribution.sample()
                    if weight.sum() != 0:
                        break
            else:
                raise ValueError('No support method RLW with {}'.format(dist))
        if dist in ['Uniform', 'Normal', 'Dirichlet']:
            batch_weight = F.softmax(weight, dim=-1).to(self.device)
        elif dist in ['Bernoulli']:
            batch_weight = weight.to(self.device)
        loss = torch.mul(losses, batch_weight).sum()
        loss.backward()
        return batch_weight.detach().cpu().numpy()