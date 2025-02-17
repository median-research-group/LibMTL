import torch, os, copy
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cvxpy as cp

from LibMTL._record import _PerformanceMeter
from LibMTL.utils import count_parameters, set_param
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method

class Trainer(nn.Module):
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
    def __init__(self, task_dict, weighting, architecture, encoder_class, decoders, 
                 rep_grad, multi_input, optim_param, scheduler_param, bilevel_param,
                 save_path=None, load_path=None, **kwargs):
        super(Trainer, self).__init__()
        
        self.device = torch.device('cuda:0')
        self.kwargs = kwargs
        self.task_dict = task_dict
        self.task_num = len(task_dict)
        self.task_name = list(task_dict.keys())
        self.rep_grad = rep_grad
        self.multi_input = multi_input
        self.scheduler_param = scheduler_param
        self.save_path = save_path
        self.load_path = load_path
        self.bilevel_param = bilevel_param
        self.weighting = weighting

        self._prepare_model(weighting, architecture, encoder_class, decoders)
        self._prepare_optimizer(optim_param, scheduler_param)
        
        self.meter = _PerformanceMeter(self.task_dict, self.multi_input)

        if self.bilevel_param is not None:
            self._prepare_UL(bilevel_param['UL_lr'])
        
    def _prepare_model(self, weighting, architecture, encoder_class, decoders):

        weighting_class = weighting_method.__dict__[weighting] 
        architecture_class = architecture_method.__dict__[architecture]
        
        class MTLmodel(architecture_class, weighting_class):
            def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, kwargs):
                super(MTLmodel, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
                self.init_param()
                
        self.model = MTLmodel(task_name=self.task_name, 
                              encoder_class=encoder_class, 
                              decoders=decoders, 
                              rep_grad=self.rep_grad, 
                              multi_input=self.multi_input,
                              device=self.device,
                              kwargs=self.kwargs['arch_args']).to(self.device)
        if self.load_path is not None:
            if os.path.isdir(self.load_path):
                self.load_path = os.path.join(self.load_path, 'best.pt')
            self.model.load_state_dict(torch.load(self.load_path), strict=False)
            print('Load Model from - {}'.format(self.load_path))
        count_parameters(self.model)
        
    def _prepare_optimizer(self, optim_param, scheduler_param):
        optim_dict = {
                'sgd': torch.optim.SGD,
                'adam': torch.optim.Adam,
                'adagrad': torch.optim.Adagrad,
                'rmsprop': torch.optim.RMSprop,
            }
        scheduler_dict = {
                'exp': torch.optim.lr_scheduler.ExponentialLR,
                'step': torch.optim.lr_scheduler.StepLR,
                'cos': torch.optim.lr_scheduler.CosineAnnealingLR,
                'reduce': torch.optim.lr_scheduler.ReduceLROnPlateau,
            }
        optim_arg = {k: v for k, v in optim_param.items() if k != 'optim'}
        self.optimizer = optim_dict[optim_param['optim']](self.model.parameters(), **optim_arg)
        if scheduler_param is not None:
            scheduler_arg = {k: v for k, v in scheduler_param.items() if k != 'scheduler'}
            self.scheduler = scheduler_dict[scheduler_param['scheduler']](self.optimizer, **scheduler_arg)
        else:
            self.scheduler = None

    def _process_data(self, loader):
        try:
            data, label = next(loader[1])
        except:
            loader[1] = iter(loader[0])
            data, label = next(loader[1])
        data = data.to(self.device, non_blocking=True)
        if not self.multi_input:
            for task in self.task_name:
                label[task] = label[task].to(self.device, non_blocking=True)
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
                train_losses[tn] = self.meter.losses[task]._update_loss(preds[task], gts[task])
        else:
            train_losses = self.meter.losses[task_name]._update_loss(preds, gts)
        return train_losses
        
    def _prepare_dataloaders(self, dataloaders):
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

    def forward4loss(self, model, inputs, gts, return_preds=False):
        if not self.multi_input:
            preds = model(inputs)
            preds = self.process_preds(preds)
            losses = self._compute_loss(preds, gts)
        else:
            losses = torch.zeros(self.task_num).to(self.device)
            preds = {}
            for tn, task in enumerate(self.task_name):
                inputs_t, gts_t = inputs[task], gts[task]
                preds_t = model(inputs_t, task)
                preds_t = preds_t[task]
                preds_t = self.process_preds(preds_t, task)
                losses[tn] = self._compute_loss(preds_t, gts_t, task)
                if return_preds:
                    preds[task] = preds_t
        if return_preds:
            return losses, preds
        else:
            return losses

    def train(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False, **kwargs):
        if self.bilevel_param is None:
            train_func = self.train_singlelevel
        else:
            train_func = eval('self.train_{}'.format(self.bilevel_param['bilevel_method']))
        train_func(train_dataloaders, test_dataloaders, epochs, 
            val_dataloaders, return_weight, **kwargs)

    def train_singlelevel(self, train_dataloaders, test_dataloaders, epochs, 
              val_dataloaders=None, return_weight=False):
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
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                train_losses = []
                for sample_num in range(3 if self.weighting in ['MoDo', 'SDMGrad'] else 1):
                    if not self.multi_input:
                        train_inputs, train_gts = self._process_data(train_loader)
                    else:
                        train_inputs, train_gts = {}, {}
                        for tn, task in enumerate(self.task_name):
                            train_input, train_gt = self._process_data(train_loader[task])
                            train_inputs[task], train_gts[task] = train_input, train_gt

                    train_losses_, train_preds = self.forward4loss(self.model, train_inputs, train_gts, return_preds=True)
                    train_losses.append(train_losses_)
                train_losses = torch.stack(train_losses).squeeze(0)

                if not self.multi_input:
                    self.meter.update(train_preds, train_gts)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts[task], task)

                self.optimizer.zero_grad(set_to_none=False)
                w = self.model.backward(train_losses, **self.kwargs['weight_args'])
                if w is not None:
                    self.batch_weight[:, epoch, batch_index] = w
                self.optimizer.step()

                if self.weighting == 'FAMO':
                    with torch.no_grad():
                        new_train_losses = self.forward4loss(self.model, train_inputs, train_gts, return_preds=False)
                        self.model.update_w(new_train_losses.detach())
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def test(self, test_dataloaders, epoch=None, mode='test', return_improvement=False):
        r'''The test process of multi-task learning.

        Args:
            test_dataloaders (dict or torch.utils.data.DataLoader): If ``multi_input`` is ``True``, \
                            it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                            dataloader which returns data and a dictionary of name-label pairs in each iteration.
            epoch (int, default=None): The current epoch. 
        '''
        test_loader, test_batch = self._prepare_dataloaders(test_dataloaders)
        
        self.model.eval()
        self.meter.record_time('begin')
        with torch.no_grad():
            if not self.multi_input:
                for batch_index in range(test_batch):
                    test_inputs, test_gts = self._process_data(test_loader)
                    test_preds = self.model(test_inputs)
                    test_preds = self.process_preds(test_preds)
                    test_losses = self._compute_loss(test_preds, test_gts)
                    self.meter.update(test_preds, test_gts)
            else:
                for tn, task in enumerate(self.task_name):
                    for batch_index in range(test_batch[tn]):
                        test_input, test_gt = self._process_data(test_loader[task])
                        test_pred = self.model(test_input, task)
                        test_pred = test_pred[task]
                        test_pred = self.process_preds(test_pred)
                        test_loss = self._compute_loss(test_pred, test_gt, task)
                        self.meter.update(test_pred, test_gt, task)
        self.meter.record_time('end')
        self.meter.get_score()
        self.meter.display(epoch=epoch, mode=mode)
        improvement = self.meter.improvement
        self.meter.reinit()
        if return_improvement:
            return improvement


    # for bilevel methods
    def _prepare_UL(self, UL_lr):
        class UL_alpha(nn.Module):
            def __init__(self, task_num):
                super(UL_alpha, self).__init__()
                self.loss_combine = nn.Parameter(torch.FloatTensor(task_num))
                
                nn.init.constant_(self.loss_combine, 1/task_num)
                
            def forward(self, loss):
                self.weight = torch.softmax(self.loss_combine, dim=0)
                final_loss = torch.sum(torch.mul(self.weight, loss))
                return final_loss
        self.ul_alpha = UL_alpha(self.task_num).to(self.device)

        self.ul_alpha_optimizer = torch.optim.Adam(self.ul_alpha.parameters(), lr=UL_lr)


    def train_MOML(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False, org_train_dataloaders=None):
        r"""MOML

            This method is proposed in `Multi-Objective Meta Learning (NeurIPS 2021) <https://proceedings.neurips.cc/paper/2021/hash/b23975176653284f1f7356ba5539cfcb-Abstract.html>`_.

            org_train_dataloaders: for QM9 only
        """

        LL_lr, LL_step = self.bilevel_param['LL_lr'], self.bilevel_param['LL_step']
        assert self.weighting == 'MGDA', 'MOML use MGDA to aggregate the gradient in UL.'
        assert LL_step == 1, "This is a special implementation of MOML, which is fast but only supports the case of LL_step=1."

        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        train_batch = int(train_batch / 2)

        if org_train_dataloaders is not None:
            org_train_loader, _ = self._prepare_dataloaders(org_train_dataloaders)

        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                # get LL, UL data
                if not self.multi_input:
                    train_inputs_LL, train_gts_LL = self._process_data(train_loader)
                    train_inputs_UL, train_gts_UL = self._process_data(train_loader)
                else:
                    train_inputs_LL, train_inputs_UL = {}, {}
                    train_gts_LL, train_gts_UL = {}, {}
                    for tn, task in enumerate(self.task_name):
                        train_inputs_LL[task], train_gts_LL[task] = self._process_data(train_loader[task])
                        train_inputs_UL[task], train_gts_UL[task] = self._process_data(train_loader[task])

                # compute LL gradient g_f (tn x d)
                meta_model = copy.deepcopy(self.model)
                losses = self.forward4loss(meta_model, train_inputs_LL, train_gts_LL)
                g_f = []
                for tn in range(self.task_num):
                    g_f_tn = torch.autograd.grad(losses[tn], meta_model.parameters(), retain_graph=True)
                    g_f.append(torch.cat([g.view(-1) for g in g_f_tn]))
                g_f = torch.stack(g_f)

                # update w* by one-step SGD
                beg = 0
                alpha = self.ul_alpha.loss_combine.data.clone()
                for p in meta_model.parameters():
                    p_grad = (F.softmax(alpha, dim=-1)@g_f[:, beg:beg+p.numel()]).view(p.size())
                    p.data -= LL_lr * p_grad
                    beg += p.numel()

                # compute softmax gradient
                g_s = torch.autograd.functional.jacobian(lambda x:F.softmax(x, dim=-1), alpha)

                # compute UL gradient g_F
                losses = self.forward4loss(meta_model, train_inputs_UL, train_gts_UL)
                alpha_grad = torch.zeros(self.task_num, len(alpha)).to(self.device)
                for tn in range(self.task_num):
                    g_F_tn = torch.autograd.grad(losses[tn], meta_model.parameters(), retain_graph=True)
                    g_F_tn = torch.cat([g.view(-1) for g in g_F_tn]) # 1 x d
                    alpha_grad[tn] = - LL_lr * g_F_tn @ g_f.t() @ g_s
                loss_data = torch.tensor([loss.item() for loss in losses]).to(self.device)
                alpha_grad = self.model.aggregated_grad(alpha_grad, loss_data, **self.kwargs['weight_args']) # --weighting MGDA
                del meta_model

                self.ul_alpha_optimizer.zero_grad()
                for param in self.ul_alpha.parameters():
                    param.grad = alpha_grad
                self.ul_alpha_optimizer.step()

                # update model
                if org_train_dataloaders is None:
                    if not self.multi_input:
                        train_inputs_all = torch.cat([train_inputs_LL, train_inputs_UL], dim=0)
                        train_gts_all = {}
                        for tn, task in enumerate(self.task_name):
                            train_gts_all[task] = torch.cat([train_gts_LL[task], train_gts_UL[task]], dim=0)
                    else:
                        train_inputs_all, train_gts_all = {}, {}
                        for tn, task in enumerate(self.task_name):
                            train_inputs_all[task] = torch.cat([train_inputs_LL[task], train_inputs_UL[task]], dim=0)
                            train_gts_all[task] = torch.cat([train_gts_LL[task], train_gts_UL[task]], dim=0)
                else:
                    train_inputs_all, train_gts_all = self._process_data(org_train_loader)

                loss, train_preds = self.forward4loss(self.model, train_inputs_all, train_gts_all, return_preds=True)
                if not self.multi_input:
                    self.meter.update(train_preds, train_gts_all)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts_all[task], task)

                loss = self.ul_alpha(loss)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ul_alpha_optimizer.zero_grad()

            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight


    def train_FORUM(self, train_dataloaders, test_dataloaders, 
                epochs, val_dataloaders=None, return_weight=False):
        r'''FORUM
    
        This method is proposed in `A First-Order Multi-Gradient Algorithm for Multi-Objective Bi-Level Optimization (ECAI 2024) <https://ebooks.iospress.nl/doi/10.3233/FAIA240793>`_.        
        '''
        LL_lr, LL_step = self.bilevel_param['LL_lr'], self.bilevel_param['LL_step']
        rho = self.bilevel_param['FORUM_rho']

        grad_index = []
        for param in self.model.parameters():
            grad_index.append(param.data.numel())
                
        train_loader, train_batch = self._prepare_dataloaders(train_dataloaders)
        train_batch = max(train_batch) if self.multi_input else train_batch
        train_batch = int(train_batch / 2)

        lambda_buffer = np.zeros([self.task_num])
        
        self.batch_weight = np.zeros([self.task_num, epochs, train_batch])
        self.model.train_loss_buffer = np.zeros([self.task_num, epochs])
        self.model.epochs = epochs
        for epoch in range(epochs):
            self.model.epoch = epoch
            self.model.train()
            self.ul_alpha.train()
            self.meter.record_time('begin')
            for batch_index in range(train_batch):
                if not self.multi_input:
                    train_inputs_LL, train_gts_LL = self._process_data(train_loader)
                    train_inputs_UL, train_gts_UL = self._process_data(train_loader)
                else:
                    train_inputs_LL, train_inputs_UL = {}, {}
                    train_gts_LL, train_gts_UL = {}, {}
                    for tn, task in enumerate(self.task_name):
                        train_inputs_LL[task], train_gts_LL[task] = self._process_data(train_loader[task])
                        train_inputs_UL[task], train_gts_UL[task] = self._process_data(train_loader[task])

                # LL f(alpha, omega)
                train_losses, train_preds = self.forward4loss(self.model, train_inputs_LL, train_gts_LL, return_preds=True)
                if not self.multi_input:
                    self.meter.update(train_preds, train_gts_LL)
                else:
                    for tn, task in enumerate(self.task_name):
                        self.meter.update(train_preds[task], train_gts_LL[task], task)
                loss = self.ul_alpha(train_losses)
                # grad from f(alpha, omega)
                g_f = torch.autograd.grad(loss, list(self.ul_alpha.parameters())+list(self.model.parameters()))
                g_f = torch.cat([g.view(-1) for g in g_f])

                # f(alpha, omega^T)
                inner_model = copy.deepcopy(self.model)
                inner_optim = torch.optim.SGD(inner_model.parameters(), lr=LL_lr, weight_decay=0)
                for i in range(LL_step):
                    train_losses = self.forward4loss(inner_model, train_inputs_LL, train_gts_LL)
                    loss = self.ul_alpha(train_losses)
                    if loss.item() > 1e+5:
                        break
                    inner_optim.zero_grad()
                    loss.backward()
                    inner_optim.step()
                    self.ul_alpha_optimizer.zero_grad()

                # grad from f^hat(alpha, omega^T)
                train_losses = self.forward4loss(inner_model, train_inputs_LL, train_gts_LL)
                loss = self.ul_alpha(train_losses)
                g_f_hat_alpha = torch.autograd.grad(loss, self.ul_alpha.parameters())
                g_f_hat_alpha = torch.cat([g.view(-1) for g in g_f_hat_alpha])

                # grad from q_beta
                g_q_beta = copy.deepcopy(g_f)
                g_q_beta[:self.task_num] = g_q_beta[:self.task_num] - g_f_hat_alpha # size: [d]

                # F(omega)
                train_losses = self.forward4loss(self.model, train_inputs_UL, train_gts_UL)
                g_F_omega_list = []
                for tn, task in enumerate(self.task_name):
                    g_F_omega_tn = torch.autograd.grad(train_losses[tn], self.model.parameters(), retain_graph=True)
                    g_F_omega_list.append(torch.cat([g.view(-1) for g in g_F_omega_tn]))
                g_F_omega_list = torch.stack(g_F_omega_list)
                gn = g_F_omega_list.pow(2).sum(-1).sqrt()
                g_F_omega_list = g_F_omega_list/(1e-8+gn.unsqueeze(1).repeat(1, g_F_omega_list.size()[1]))

                g_F_omega_list = torch.cat([torch.zeros(self.task_num, self.task_num).to(self.device), g_F_omega_list], dim=1)

                # pi
                pi = []
                for tn in range(self.task_num):
                    pi.append(rho - ((g_q_beta*g_F_omega_list[tn]).sum()/(g_q_beta.norm().pow(2)+1e-8)).item())

                w_constant = rho * (g_q_beta.norm().pow(2)+1e-8).item()

                # A
                A = torch.cat([g_F_omega_list, g_q_beta.unsqueeze(0)], dim=0) # (task_num+1) x d
                AAT = (A @ A.t()).detach().cpu().numpy()

                # solve problem (14)
                c, v = np.linalg.eig(AAT)
                gg_sqrt = v @ np.diag(np.sqrt(np.maximum(c,0))) @ np.linalg.inv(v)
                g_cp = cp.Parameter(shape=(self.task_num+1, self.task_num+1), value=gg_sqrt)
                w_cp = cp.Variable(shape=(self.task_num+1), nonneg=True)
                constraints = [cp.sum(w_cp[:-1]) == 1, 
                               w_cp[-1] >= cp.sum([w_cp[tn]*pi[tn] for tn in range(self.task_num)]), 
                               w_cp >= 0]
                prob = cp.Problem(cp.Minimize(cp.quad_over_lin(g_cp @ w_cp, 1) - w_cp[-1] * w_constant), constraints)
                prob.solve()
                w_cpu = w_cp.value

                # EMA lambda
                for tn in range(self.task_num):
                    lambda_buffer[tn] = lambda_buffer[tn] + (1 / (epoch+1)**(3/4)) * (w_cpu[tn] - lambda_buffer[tn])

                # compute problem (12)
                nu = max(sum([lambda_buffer[tn]*pi[tn] for tn in range(self.task_num)]), 0)
                g_final = sum([lambda_buffer[tn]*g_F_omega_list[tn] for tn in range(self.task_num)]) + nu*g_q_beta

                self.ul_alpha_optimizer.zero_grad()
                for param in self.ul_alpha.parameters():
                    param.grad = g_final[:self.task_num]

                self.optimizer.zero_grad()
                count = 0
                for param in self.model.parameters():
                    beg = 0 if count == 0 else sum(grad_index[:count])
                    end = sum(grad_index[:(count+1)])
                    param.grad = g_final[self.task_num:][beg:end].contiguous().view(param.data.size()).data.clone()
                    count += 1

                self.ul_alpha_optimizer.step()
                self.optimizer.step()
            
            self.meter.record_time('end')
            self.meter.get_score()
            self.model.train_loss_buffer[:, epoch] = self.meter.loss_item
            self.meter.display(epoch=epoch, mode='train')
            self.meter.reinit()
            
            if val_dataloaders is not None:
                self.meter.has_val = True
                val_improvement = self.test(val_dataloaders, epoch, mode='val', return_improvement=True)
            self.test(test_dataloaders, epoch, mode='test')
            if self.scheduler is not None:
                if self.scheduler_param['scheduler'] == 'reduce' and val_dataloaders is not None:
                    self.scheduler.step(val_improvement)
                else:
                    self.scheduler.step()
            if self.save_path is not None and self.meter.best_result['epoch'] == epoch:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'best.pt'))
                print('Save Model {} to {}'.format(epoch, os.path.join(self.save_path, 'best.pt')))
        self.meter.display_best_result()
        if return_weight:
            return self.batch_weight