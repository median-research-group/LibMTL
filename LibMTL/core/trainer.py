import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Type

from LibMTL.utils.recorder import ExpRecorder
from LibMTL.utils.timer import TimeRecorder
from LibMTL.utils.builder import build_from_cfg, build_dataloader
from LibMTL.core.common_init import common_init
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

class Trainer:
    r'''A Multi-Task Learning Trainer.

    This is a unified and extensible training framework for multi-task learning. 

    Args:
        task_dict (dict): A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                            The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                            **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                            The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metrics \
                            for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                            functions, meaning how to update thoes objectives in the training process and obtain the final \
                            scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                            metric. The list of **weight** has ``m`` binary integers corresponding to each \
                            metric, where ``1`` means the higher the score is, the better the performance, \
                            ``0`` means the opposite.                           
        weighting (class): A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
        architecture (class): An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
        encoder_class (class): A neural network class.
        decoders (dict): A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
        rep_grad (bool): If ``True``, the gradient of the representation for each task can be computed.
        multi_input (bool): Is ``True`` if each task has its own input data, otherwise is ``False``. 
        optim_param (dict): A dictionary of configurations for the optimizier.
        scheduler_param (dict): A dictionary of configurations for learning rate scheduler. \
                                 Set it to ``None`` if you do not use a learning rate scheduler.
        kwargs (dict): A dictionary of hyperparameters of weighting and architecture methods.

    .. note::
            It is recommended to use :func:`LibMTL.config.prepare_args` to return the dictionaries of ``optim_param``, \
            ``scheduler_param``, and ``kwargs``.

    Examples::
        
        import torch.nn as nn
        from LibMTL import Trainer
        from LibMTL.loss import CE_loss_fn
        from LibMTL.metrics import acc_update_fun, acc_score_fun
        from LibMTL.weighting import EW
        from LibMTL.architecture import HPS
        from LibMTL.model import ResNet18
        from LibMTL.config import prepare_args

        task_dict = {'A': {'metrics': ['Acc'],
                           'metrics_fn': [acc_update_fun, acc_score_fun],
                           'loss_fn': [CE_loss_fn],
                           'weight': [1]}}
        
        decoders = {'A': nn.Linear(512, 31)}
        
        # You can use command-line arguments and return configurations by ``prepare_args``.
        # kwargs, optim_param, scheduler_param = prepare_args(params)
        optim_param = {'optim': 'adam', 'lr': 1e-3, 'weight_decay': 1e-4}
        scheduler_param = {'scheduler': 'step'}
        kwargs = {'weight_args': {}, 'arch_args': {}}

        trainer = Trainer(task_dict=task_dict,
                          weighting=EW,
                          architecture=HPS,
                          encoder_class=ResNet18,
                          decoders=decoders,
                          rep_grad=False,
                          multi_input=False,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)

    '''
    def __init__(self, 
                 cfg_file: str,
                 task_dict: Optional[dict] = None, 
                 encoder_class: Optional[Type[nn.Module]] = None,
                 decoders: Optional[nn.ModuleDict] = None,
                 dataloaders: Optional[dict] = None,
                 **kwargs):
        
        self.kwargs = kwargs

        self.cfg, self.saver, self.device = common_init(cfg_file)
        self.use_gpu = True if torch.cuda.is_available() else False
        ## tasks
        self.task_dict = self.cfg['tasks'] if task_dict is None else task_dict
        self.task_num = len(self.task_dict['info'])
        self.task_name = list(self.task_dict['info'].keys())
        self.multi_input = self.task_dict['multi_input']
        ## train config
        self.epochs = self.cfg['training']['epochs']

        self._prepare_model() # self.model
        self._prepare_optimizer() # self.optimizer, self.scheduler
        if dataloaders is None:
            self._prepare_dataloaders() # self.dataloaders
        else:
            self.dataloaders = dataloaders
        self._prepare_loss() # self.losses
        self._prepare_metric() # self.metrics
        self.recorder = ExpRecorder(self.task_dict, self.losses, self.metrics)
        
    def _prepare_model(self):

        architecture = architecture_method.__dict__[self.cfg['architecture']['name']]
        weighting = weighting_method.__dict__[self.cfg['weighting']['name']]
        arch_arg = {k: v for k, v in self.cfg['architecture'].items() if k != 'name'}
        def encoder_class():
            return build_from_cfg(build_type='encoder', cfg=self.cfg['model']['encoder'])
        decoders = nn.ModuleDict({})
        for task in self.task_name:
            decoders[task] = build_from_cfg(build_type='decoder', cfg=self.cfg['model']['decoders'][task])
        
        class MTLmodel(architecture, weighting):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.cfg['weighting']['rep_grad'], 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=arch_arg).to(self.device)
        
    def _prepare_optimizer(self):
        self.optimizer = build_from_cfg(build_type='optimizer',
                                        cfg=self.cfg['training']['optimizer'],
                                        other_arg={'params': self.model.parameters()})
        if 'scheduler' in list(self.cfg['training'].keys()):
            self.scheduler = build_from_cfg(build_type='lr_scheduler',
                                            cfg=self.cfg['training']['lr_scheduler'],
                                            other_arg={'optimizer': self.optimizer})
        else:
            self.scheduler = None

    def _prepare_dataloaders(self):
        self.dataloaders = {}
        for mode in list(self.cfg['training']['dataloader'].keys()):
            if mode in ['train', 'val', 'test']:
                if self.multi_input:
                    dataset = {}
                    for task in self.task_name:
                        dataset[task] = build_from_cfg(build_type='dataset',
                                            cfg=self.cfg['training']['dataloader'][mode]['dataset'],
                                            other_arg={'mode': mode, 'current_task': task})
                else:
                    dataset = build_from_cfg(build_type='dataset',
                                        cfg=self.cfg['training']['dataloader'][mode]['dataset'],
                                        other_arg={'task_name': self.task_name})
                dataloader_arg = {k: v for k, v in self.cfg['training']['dataloader'][mode].items() if k not in  ['dataset']}
                self.dataloaders[mode] = build_dataloader(dataset, self.multi_input, self.task_name, **dataloader_arg)

    def _prepare_loss(self):
        self.losses = {}
        for task in self.task_name:
            self.losses[task] = build_from_cfg(build_type='loss',
                                            cfg=self.cfg['tasks']['info'][task]['loss_fn'])

    def _prepare_metric(self):
        self.metrics = {}
        for task in self.task_name:
            self.metrics[task] = build_from_cfg(build_type='metric',
                                            cfg=self.cfg['tasks']['info'][task]['metric_fn'])

    def data_step(self, loader):
        try:
            data, label = loader[1].next()
        except:
            loader[1] = iter(loader[0])
            data, label = loader[1].next()
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input and isinstance(label, dict):
            for k, v in label.items():
                label[k] = v.to(self.device, non_blocking=True)
        else:
            label = label.to(self.device, non_blocking=True)
        return data, label
    
    def process_preds(self, preds, task_name=None):
        r'''The processing of prediction for each task. 

        - The default is no processing. If necessary, you can rewrite this function. 
        - If ``multi_input`` is ``True``, ``task_name`` is valid and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
        - otherwise, ``task_name`` is invalid and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

        Args:
            preds (dict or torch.Tensor): The prediction of ``task_name`` or all tasks.
            task_name (str): The string of task name.
        '''
        return preds

    def _compute_loss(self, preds, gts, task_name=None):
        if not self.multi_input:
            train_losses = torch.zeros(self.task_num).to(self.device)
            for tn, task in enumerate(self.task_name):
                train_losses[tn] = self.losses[task].update_loss(preds[task], gts[task])
        else:
            train_losses = self.losses[task_name].update_loss(preds[task_name], gts)
        return train_losses
        
    def _process_dataloaders(self, dataloaders):
        if not self.multi_input:
            loader = [dataloaders, iter(dataloaders)]
            return loader, len(dataloaders)
        else:
            loader = {}
            batch_num = []
            for task in self.task_name:
                loader[task] = [dataloaders[task], iter(dataloaders[task])]
                batch_num.append(len(dataloaders[task]))
            return loader, batch_num

    def run_step(self, dataloader, task_name=None):
        with TimeRecorder(self.use_gpu) as data_time_t:
            inputs, gts = self.data_step(dataloader)
        self.data_time += data_time_t.elapse
        with TimeRecorder(self.use_gpu) as forward_time_t:
            preds = self.model(inputs, task_name)
        self.forward_time += forward_time_t.elapse
        preds = self.process_preds(preds)
        return gts, preds

    def train(self, return_weight=False):
        r'''The training process of multi-task learning.

        Args:
            train_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for training. \
                            If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                            Otherwise, it is a single dataloader which returns data and a dictionary \
                            of name-label pairs in each iteration.

            test_dataloaders (dict or torch.utils.data.DataLoader): The dataloaders used for the validation or testing. \
                            The same structure with ``train_dataloaders``.
            epochs (int): The total training epochs.
            return_weight (bool): if ``True``, the loss weights will be returned.
        '''
        train_loader, train_batch = self._process_dataloaders(self.dataloaders['train'])
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, self.epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, self.epochs])
        for epoch in range(self.epochs):
            self.data_time, self.forward_time, self.backward_time = 0, 0, 0
            self.model.epoch = epoch
            self.model.train()
            self.recorder.reset(epoch=epoch, mode='train')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_gts, train_preds = self.run_step(train_loader)
                    train_losses = self._compute_loss(train_preds, train_gts)
                    self.recorder.update(train_preds, train_gts)
                else:
                    train_losses = torch.zeros(self.task_num).to(self.device)
                    for tn, task in enumerate(self.task_name):
                        train_gt, train_pred = self.run_step(train_loader[task], task)
                        train_losses[tn] = self._compute_loss(train_pred, train_gt, task)
                        self.recorder.update(train_pred[task], train_gt, task)

                self.optimizer.zero_grad()
                weighting_arg = {k: v for k, v in self.cfg['weighting'].items() if k not in  ['name', 'rep_grad']}
                with TimeRecorder(self.use_gpu) as backward_time_t:
                    w = self.model.backward(train_losses, **weighting_arg)
                self.backward_time += backward_time_t.elapse
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()
            
            self.recorder.record_time(self.data_time, self.forward_time, self.backward_time)
            self.recorder.get_score()
            self.recorder.print_result()
            self.model.train_loss_buffer[:, epoch] = self.recorder._return_loss(epoch, mode='train')
            self.saver.save_best_model(self.model, self.recorder.best_epoch)
            
            if 'val' in list(self.dataloaders.keys()):
                self.recorder.has_val = True
                self.evaluate(self.dataloaders['val'], epoch, mode='val')
            self.evaluate(self.dataloaders['test'], epoch, mode='test')
            if self.scheduler is not None:
                self.scheduler.step()
        self.recorder.print_best_result()
        if return_weight:
            return self.batch_weight


    def evaluate(self, test_dataloaders, epoch=None, mode='test'):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._process_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.recorder.reset(epoch=epoch, mode=mode)
        self.data_time, self.forward_time = 0, 0
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_gts, test_preds = self.run_step(test_loader)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.recorder.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_gt, test_pred = self.run_step(test_loader[task], task)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.recorder.update(test_pred[task], test_gt, task)
        self.recorder.record_time(self.data_time, self.forward_time)
        self.recorder.get_score()
        self.recorder.print_result()
