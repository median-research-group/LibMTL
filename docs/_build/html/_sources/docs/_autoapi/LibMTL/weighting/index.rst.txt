:mod:`LibMTL.weighting`
=======================

.. py:module:: LibMTL.weighting






.. py:class:: AbsWeighting

   Bases: :py:obj:`torch.nn.Module`

   An abstract class for weighting strategies.


   .. py:method:: init_param(self)

      Define and initialize some trainable parameters required by specific weighting methods.



   .. py:method:: backward(self, losses, **kwargs)
      :property:

      :param losses: A list of loss of each task.
      :type losses: list
      :param kwargs: A dictionary of hyperparameters of weighting methods.
      :type kwargs: dict



.. py:class:: EW

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Equally Weighting (EW).

   The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` means the number of tasks.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: GradNorm

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Gradient Normalization (GradNorm).

   This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
   and implemented by us.

   :param alpha: The strength of the restoring force which pulls tasks back to a common training rate.
   :type alpha: float, default=1.5

   .. py:method:: init_param(self)


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: MGDA

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Multiple Gradient Descent Algorithm (MGDA).

   This method is proposed in `Multi-Task Learning as Multi-Objective Optimization (NeurIPS 2018) <https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/isl-org/MultiObjectiveOptimization>`_.

   :param mgda_gn: The type of gradient normalization.
   :type mgda_gn: {'none', 'l2', 'loss', 'loss+'}, default='none'

   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: UW

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Uncertainty Weights (UW).

   This method is proposed in `Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR 2018) <https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf>`_ \
   and implemented by us.


   .. py:method:: init_param(self)


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: DWA

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Dynamic Weight Average (DWA).

   This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

   :param T: The softmax temperature.
   :type T: float, default=2.0

   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: GLS

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Geometric Loss Strategy (GLS).

   This method is proposed in `MultiNet++: Multi-Stream Feature Aggregation and Geometric Loss Strategy for Multi-Task Learning (CVPR 2019 workshop) <https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf>`_ \
   and implemented by us.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: GradDrop

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Gradient Sign Dropout (GradDrop).

   This method is proposed in `Just Pick a Sign: Optimizing Deep Multitask Models with Gradient Sign Dropout (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html>`_ \
   and implemented by us.

   :param leak: The leak parameter for the weighting matrix.
   :type leak: float, default=0.0

   .. warning::
           GradDrop is not supported with parameter gradients, i.e., ``rep_grad`` must be ``True``.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: PCGrad

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Project Conflicting Gradients (PCGrad).

   This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
   and implemented by us.

   .. warning::
           PCGrad is not supported with representation gradients, i.e., ``rep_grad`` must be ``False``.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: GradVac

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Gradient Vaccine (GradVac).

   This method is proposed in `Gradient Vaccine: Investigating and Improving Multi-task Optimization in Massively Multilingual Models (ICLR 2021 Spotlight) <https://openreview.net/forum?id=F1vEjWK-lH_>`_ \
   and implemented by us.

   :param beta: The exponential moving average (EMA) decay parameter.
   :type beta: float, default=0.5

   .. warning::
           GradVac is not supported with representation gradients, i.e., ``rep_grad`` must be ``False``.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: IMTL

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Impartial Multi-task Learning (IMTL).

   This method is proposed in `Towards Impartial Multi-task Learning (ICLR 2021) <https://openreview.net/forum?id=IMPnRXEWpvr>`_ \
   and implemented by us.


   .. py:method:: init_param(self)


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: CAGrad

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Conflict-Averse Gradient descent (CAGrad).

   This method is proposed in `Conflict-Averse Gradient Descent for Multi-task learning (NeurIPS 2021) <https://openreview.net/forum?id=_61Qh8tULj_>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/Cranial-XIX/CAGrad>`_.

   :param calpha: A hyperparameter that controls the convergence rate.
   :type calpha: float, default=0.5
   :param rescale: The type of gradient rescale.
   :type rescale: {0, 1, 2}, default=1

   .. warning::
           CAGrad is not supported with representation gradients, i.e., ``rep_grad`` must be ``False``.


   .. py:method:: backward(self, losses, **kwargs)



.. py:class:: RLW

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Random Loss Weighting (RLW).

   This method is proposed in `A Closer Look at Loss Weighting in Multi-Task Learning (arXiv:2111.10603) <https://arxiv.org/abs/2111.10603>`_ \
   and implemented by us.

   :param dist: The type of distribution where the loss weigghts are sampled from.
   :type dist: {'Uniform', 'Normal', 'Dirichlet', 'Bernoulli', 'constrained_Bernoulli'}, default='Normal'

   .. py:method:: backward(self, losses, **kwargs)

      :param losses: A list of loss of each task.
      :type losses: list
      :param kwargs: A dictionary of hyperparameters of weighting methods.
      :type kwargs: dict




