import random, torch, os
import numpy as np
import torch.nn as nn

def set_random_seed(seed: int, deterministic: bool = False):
    r"""Set the random seed for reproducibility.

    Args:
        seed (int, default=0): The random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        
def set_device(gpu_id):
    r"""Set the device where model and data will be allocated. 

    Args:
        gpu_id (str, default='0'): The id of gpu.
    """
    if torch.cuda.is_available():   
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device('cuda:{}'.format(gpu_id))
        device = torch.device('cuda:{}'.format(gpu_id))
    else:
        device = torch.device('cpu')
    return device

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
        
def count_improvement(base_result: dict, 
                      new_result: dict, 
                      weight: dict) -> float:
    r"""Calculate the improvement between two results as

    .. math::
        \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T 
        \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{N_{t,m}}.

    Args:
        base_result (dict): A dictionary of scores of all metrics of all tasks.
        new_result (dict): The same structure with ``base_result``.
        weight (dict): The same structure with ``base_result`` while each element is binary integer representing whether higher or lower score is better.

    Returns:
        float: The improvement between ``new_result`` and ``base_result``.

    Examples::

        base_result = {'A': {'AM1': 96, 'AM2': 98}, 'B': {'BM1': 0.2}}
        new_result = {'A': {'AM1': 93, 'AM2': 99}, 'B': {'BM1': 0.5}}
        weight = {'A': {'AM1':1, 'AM2': 0}, 'B': {'BM1': 1}}

        print(count_improvement(base_result, new_result, weight))
    """
    improvement = 0
    count = 0
    for task in list(base_result.keys()):
        we_array = np.array([weight[task][metric] for metric in list(base_result[task].keys())])
        br_array = np.array([base_result[task][metric] for metric in list(base_result[task].keys())])
        nr_array = np.array([new_result[task][metric] for metric in list(base_result[task].keys())])
        improvement += (((-1) ** we_array) * (br_array - nr_array) / br_array).mean()
        count += 1
    return improvement / count