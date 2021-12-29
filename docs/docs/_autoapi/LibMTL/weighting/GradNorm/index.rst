:mod:`LibMTL.weighting.GradNorm`
================================

.. py:module:: LibMTL.weighting.GradNorm






.. py:class:: GradNorm

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Gradient Normalization (GradNorm).

   This method is proposed in `GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks (ICML 2018) <http://proceedings.mlr.press/v80/chen18a/chen18a.pdf>`_ \
   and implemented by us.

   :param alpha: The strength of the restoring force which pulls tasks back to a common training rate.
   :type alpha: float, default=1.5

   .. py:method:: init_param(self)


   .. py:method:: backward(self, losses, **kwargs)




