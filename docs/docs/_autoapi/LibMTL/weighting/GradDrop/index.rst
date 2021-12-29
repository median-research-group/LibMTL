:mod:`LibMTL.weighting.GradDrop`
================================

.. py:module:: LibMTL.weighting.GradDrop






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




