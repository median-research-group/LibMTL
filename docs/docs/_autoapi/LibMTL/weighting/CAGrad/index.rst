:mod:`LibMTL.weighting.CAGrad`
==============================

.. py:module:: LibMTL.weighting.CAGrad






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




