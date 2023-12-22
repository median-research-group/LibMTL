import random, torch, os
import numpy as np
import torch.nn as nn

def get_root_dir():
    r"""Return the root path of project."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def set_random_seed(seed):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
def set_device(gpu_id):
    r"""Set the device where model and data will be allocated. 

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

def count_parameters(model):
    r'''Calculate the number of parameters for a model.

    Args:
        model (torch.nn.Module): A neural network module.
    '''
    trainable_params = 0
    non_trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            trainable_params += p.numel()
        else:
            non_trainable_params += p.numel()
    print('='*40)
    print('Total Params:', trainable_params + non_trainable_params)
    print('Trainable Params:', trainable_params)
    print('Non-trainable Params:', non_trainable_params)
        
def count_improvement(base_result, new_result, weight):
    r"""Calculate the improvement between two results as

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T 
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{B_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each element is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': [96, 98], 'B': [0.2]}
        new_result = {'A': [93, 99], 'B': [0.5]}
        weight = {'A': [1, 0], 'B': [1]}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        improvement += (((-1)**np.array(weight[task]))*\
                        (np.array(base_result[task])-np.array(new_result[task]))/\
                         np.array(base_result[task])).mean()
        count += 1
    return improvement/count
