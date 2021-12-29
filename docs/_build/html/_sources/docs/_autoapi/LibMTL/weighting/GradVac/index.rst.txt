:mod:`LibMTL.weighting.GradVac`
===============================

.. py:module:: LibMTL.weighting.GradVac






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




