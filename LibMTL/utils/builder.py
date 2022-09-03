import torch, LibMTL
from typing import Dict, Union, Optional, Any
from torch.utils.data import DataLoader

def build_from_cfg(build_type: str,
                cfg: Union[str, dict], 
                other_arg: Optional[dict] = None,
                fun_key: str = 'name',
                ignore_key: Optional[Union[list, str]] = None):
    if build_type == 'optimizer':
        module = 'torch.optim.'
    elif build_type == 'lr_scheduler':
        module = 'torch.optim.lr_scheduler.'
    elif build_type == 'encoder':
        module = 'LibMTL.model.encoder.'
    elif build_type == 'decoder':
        module = 'LibMTL.model.decoder.'
    elif build_type == 'loss':
        module = 'LibMTL.loss.'
    elif build_type == 'metric':
        module = 'LibMTL.metric.'
    elif build_type == 'dataset':
        module = 'LibMTL.dataset.'
    else:
        raise ValueError("No support build_type {}. Optional types are 'optimizer', \
                         'lr_scheduler', 'encoder', 'decoder', 'loss', 'metric', \
                         and 'dataset'.".format(build_type))

    if isinstance(cfg, str):
        fun_name = cfg
        fun_arg = {}
    else:
        fun_name = cfg[fun_key]
        fun_key = [fun_key]
        if ignore_key is not None:
            if isinstance(ignore_key, str):
                fun_key.append(ignore_key)
            else:
                fun_key += ignore_key
        fun_arg = {k:v for k, v in cfg.items() if k not in fun_key}
    if other_arg is not None:
        fun_arg.update(other_arg)
    return eval(module+fun_name)(**fun_arg)


def build_dataloader(dataset,
                     multi_input: bool,
                     task_name: list,
                     **args):
    if multi_input and isinstance(dataset, dict):
        dataloader = {}
        for task in task_name:
            dataloader[task] = DataLoader(dataset[task], **args)
    else:
        dataloader = DataLoader(dataset, **args)
    return dataloader
