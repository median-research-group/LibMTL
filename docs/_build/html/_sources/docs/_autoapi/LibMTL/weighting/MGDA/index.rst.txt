:mod:`LibMTL.weighting.MGDA`
============================

.. py:module:: LibMTL.weighting.MGDA






.. py:class:: MGDA

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Multiple Gradient Descent Algorithm (MGDA).

   This method is proposed in `Multi-Task Learning as Multi-Objective Optimization (NeurIPS 2018) <https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/isl-org/MultiObjectiveOptimization>`_.

   :param mgda_gn: The type of gradient normalization.
   :type mgda_gn: {'none', 'l2', 'loss', 'loss+'}, default='none'

   .. py:method:: backward(self, losses, **kwargs)




