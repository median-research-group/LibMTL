import os, yaml, torch
from time import strftime, gmtime

from LibMTL.utils.logger import get_root_logger

class ExpSaver:
    def __init__(self, cfg: dict):
        self.save_dir = os.path.abspath(os.path.expanduser(cfg['general']['save_dir']))
        self.exp_name = cfg['general']['exp_name']
        self.old_model = -1
        self.mkdir_or_exist()
        self.logger = get_root_logger(log_file=os.path.join(self.save_dir, self.exp_name, 'log.txt'))
        self.save_cfg(cfg)
        self.logger.info(self.message + '\n' + '='*60)

    def mkdir_or_exist(self):
        self.exp_name = strftime("%Y-%m-%d@%H:%M:%S", gmtime()) + '_' + self.exp_name
        path = os.path.join(self.save_dir, self.exp_name)
        os.makedirs(path)
        self.message = 'Save path: {}'.format(path)

    def save_best_model(self, current_model, best_epoch: int):
        if self.old_model != best_epoch:
            path = os.path.join(self.save_dir, self.exp_name, 'best_model_{}.pt'.format(best_epoch))
            torch.save(current_model.state_dict(), path)
            old_path = os.path.join(self.save_dir, self.exp_name, 'best_model_{}.pt'.format(self.old_model))
            self.old_model = best_epoch

    def save_cfg(self, cfg: dict):
        path = os.path.join(self.save_dir, self.exp_name, 'config.yaml')
        if os.path.exists(path):
            os.remove(path)
        with open(path, 'w') as f:
            yaml.dump(cfg, f, sort_keys=False)
