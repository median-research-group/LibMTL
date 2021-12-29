:mod:`LibMTL`
=============

.. py:module:: LibMTL






.. py:class:: Trainer(task_dict, weighting, architecture, encoder_class, decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs)

   Bases: :py:obj:`torch.nn.Module`

   A Multi-Task Learning Trainer.

   This is a unified and extensible training framework for multi-task learning.

   :param task_dict: A dictionary of name-information pairs of type (:class:`str`, :class:`dict`). \
                     The sub-dictionary for each task has four entries whose keywords are named **metrics**, \
                     **metrics_fn**, **loss_fn**, **weight** and each of them corresponds to a :class:`list`.
                     The list of **metrics** has ``m`` strings, repersenting the name of ``m`` metric objectives \
                     for this task. The list of **metrics_fn** has two elements, i.e., the updating and score \
                     functions, meaning how to update thoes objectives in the training process and get the final \
                     scores, respectively. The list of **loss_fn** has ``m`` loss functions corresponding to each \
                     metric objective. The list of **weight** has ``m`` binary integers corresponding to each \
                     metric objective, where ``1`` means the score higher the performance of this objective better, \
                     ``0`` otherwise.
   :type task_dict: dict
   :param weighting: A weighting strategy class based on :class:`LibMTL.weighting.abstract_weighting.AbsWeighting`.
   :type weighting: class
   :param architecture: An architecture class based on :class:`LibMTL.architecture.abstract_arch.AbsArchitecture`.
   :type architecture: class
   :param encoder_class: A neural network class.
   :type encoder_class: class
   :param decoders: A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
   :type decoders: dict
   :param rep_grad: If ``True``, the gradient of the representation for each task can be computed.
   :type rep_grad: bool
   :param multi_input: Is ``True`` if each task has its own input data, ``False`` otherwise.
   :type multi_input: bool
   :param optim_param: A dictionary of configuration for the optimizier.
   :type optim_param: dict
   :param scheduler_param: A dictionary of configuration for learning rate scheduler. \
                           Setting it as ``None`` if you do not use a learning rate scheduler.
   :type scheduler_param: dict
   :param kwargs: A dictionary of hyperparameters of weighting and architecture methods.
   :type kwargs: dict

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


   .. py:method:: process_preds(self, preds, task_name=None)

      The processing of prediction for each task.

      - The default is no processing. If necessary, you can redefine this function.
      - If ``multi_input``, ``task_name`` is valid, and ``preds`` with type :class:`torch.Tensor` is the prediction of this task.
      - otherwise, ``task_name`` is invalid, and ``preds`` is a :class:`dict` of name-prediction pairs of all tasks.

      :param preds: The prediction of ``task_name`` or all tasks.
      :type preds: dict or torch.Tensor
      :param task_name: The string of task name.
      :type task_name: str


   .. py:method:: train(self, train_dataloaders, test_dataloaders, epochs, val_dataloaders=None, return_weight=False)

      The training process of multi-task learning.

      :param train_dataloaders: The dataloaders used for training. \
                                If ``multi_input`` is ``True``, it is a dictionary of name-dataloader pairs. \
                                Otherwise, it is a single dataloader which returns data and a dictionary \
                                of name-label pairs in each iteration.
      :type train_dataloaders: dict or torch.utils.data.DataLoader
      :param test_dataloaders: The dataloaders used for validation or test. \
                               The same structure with ``train_dataloaders``.
      :type test_dataloaders: dict or torch.utils.data.DataLoader
      :param epochs: The total training epochs.
      :type epochs: int
      :param return_weight: if ``True``, the loss weights will be returned.
      :type return_weight: bool


   .. py:method:: test(self, test_dataloaders, epoch=None, mode='test')

      The test process of multi-task learning.

      :param test_dataloaders: If ``multi_input`` is ``True``, \
                               it is a dictionary of name-dataloader pairs. Otherwise, it is a single \
                               dataloader which returns data and a dictionary of name-label pairs in each iteration.
      :type test_dataloaders: dict or torch.utils.data.DataLoader
      :param epoch: The current epoch.
      :type epoch: int, default=None




.. toctree::
   :titlesonly:
   :maxdepth: 1

   architecture/index.rst
   model/index.rst
   weighting/index.rst


