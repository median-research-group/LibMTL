import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional
from pprint import pformat

from LibMTL.utils.utils import count_improvement
from LibMTL.loss import *
from LibMTL.metric import *

from LibMTL.utils.logger import get_root_logger

class ExpRecorder(object):
    def __init__(self, 
                 task_dict: dict, 
                 losses: dict,
                 metrics: dict,
                 base_result: Optional[dict] = None):
        
        self.task_dict = task_dict
        self.losses = losses
        self.metrics = metrics
        self.task_name = list(self.task_dict['info'].keys())
        
        self.weight = {task: self.metrics[task].metric_info for task in self.task_name}
        self.base_result = base_result
        self.best_improvement = -1e+2
        self.best_epoch = 0

        self.all_results = {}
        
        self.has_val = False

        self.logger = get_root_logger()
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                score = self.metrics[task].score_fun()
                for k, v in score.items():
                    self.all_results[self.current_epoch][self.current_mode][task]['metrics'][k] = v
                self.all_results[self.current_epoch][self.current_mode][task]['loss'] = self.losses[task].average_loss()
        self.update_best_result()

    def _return_loss(self, epoch, mode):
        return np.array([self.all_results[epoch][mode][task]['loss'] for task in self.task_name])

    def _transform(self, result: dict) -> dict:
        transform_result = {task: result[task]['metrics'] for task in self.task_name}
        return transform_result

    def update_best_result(self):
        if self.current_mode == 'test':
            mode = 'val' if self.has_val else 'test'
            if self.base_result is None:
                base_result = self._transform(self.all_results[0][mode])
            current_result = self._transform(self.all_results[self.current_epoch][mode])
            improvement = count_improvement(base_result, current_result, self.weight)
            if improvement > self.best_improvement:
                self.best_improvement = improvement
                self.best_epoch = self.current_epoch

    def print_result(self, show_all_tasks: bool = True):
        current_result = self.all_results[self.current_epoch][self.current_mode]
        message = '[Epoch: {}] [Mode: {}]\n'.format(self.current_epoch, self.current_mode)
        self.logger.info(message+pformat(self.all_results[self.current_epoch][self.current_mode])+'\n'+'-'*60)

    def print_best_result(self):
        message = 'Best Epoch {}; Result {}'.format(self.best_epoch, 
                            self._transform(self.all_results[self.best_epoch]['test']))
        self.logger.info(message)

    def record_time(self, data_time: float, forward_time: float, backward_time: Optional[float] = None):
        self.all_results[self.current_epoch][self.current_mode]['time']['data_time'] = data_time
        self.all_results[self.current_epoch][self.current_mode]['time']['forward_time'] = forward_time
        if backward_time is not None:
            self.all_results[self.current_epoch][self.current_mode]['time']['backward_time'] = backward_time
        
    def reset(self, epoch: int, mode: str):
        for task in self.task_name:
            self.losses[task].reinit()
            self.metrics[task].reinit()
        self.current_mode = mode
        self.current_epoch = epoch
        if epoch not in list(self.all_results.keys()):
            self.all_results[epoch] = {}
        self.all_results[epoch][mode] = {'time': {'data_time': 0, 'forward_time': 0}}
        if mode == 'train':
            self.all_results[epoch]['train']['time']['backward_time'] = 0
        for task in self.task_name:
            self.all_results[epoch][mode][task] = {'loss': 0,
                                                   'metrics': {}}
            for metric_name in self.metrics[task].metric_name:
                self.all_results[epoch][mode][task]['metrics'][metric_name] = 0
