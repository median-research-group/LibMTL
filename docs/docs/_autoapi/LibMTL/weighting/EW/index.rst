:mod:`LibMTL.weighting.EW`
==========================

.. py:module:: LibMTL.weighting.EW






.. py:class:: EW

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Equally Weighting (EW).

   The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` means the number of tasks.


   .. py:method:: backward(self, losses, **kwargs)




