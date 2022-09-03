import yaml

def load_cfg(cfg_file: str) -> dict:
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)
    w_type = cfg['weighting']['name']
    if w_type in list(default_config.keys()):
        w_arg = {k: v for k, v in cfg['weighting'].items() if k not in  ['name', 'rep_grad']}
        diff = list(set(default_config[w_type].keys()).difference(set(w_arg.keys())))
        if len(diff) != 0:
            for df in diff:
                cfg['weighting'][df] = default_config[w_type][df]
    return cfg


default_config = {'DWA': {'T': 2},
                  'MGDA': {'mgda_gn': 'none'},
                  'GradVac': {'beta': 0.5},
                  'GradNorm': {'alpha': 1.5},
                  'GradDrop': {'leak': 0.0},
                  'CAGrad': {'alpha': 0.5, 'rescale': 1}}