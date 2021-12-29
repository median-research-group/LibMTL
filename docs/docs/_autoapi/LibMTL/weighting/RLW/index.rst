:mod:`LibMTL.weighting.RLW`
===========================

.. py:module:: LibMTL.weighting.RLW






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




