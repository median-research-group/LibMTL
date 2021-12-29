:mod:`LibMTL.weighting.PCGrad`
==============================

.. py:module:: LibMTL.weighting.PCGrad






.. py:class:: PCGrad

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Project Conflicting Gradients (PCGrad).

   This method is proposed in `Gradient Surgery for Multi-Task Learning (NeurIPS 2020) <https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html>`_ \
   and implemented by us.

   .. warning::
           PCGrad is not supported with representation gradients, i.e., ``rep_grad`` must be ``False``.


   .. py:method:: backward(self, losses, **kwargs)




