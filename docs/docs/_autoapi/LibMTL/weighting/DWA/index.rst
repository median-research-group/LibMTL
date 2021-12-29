:mod:`LibMTL.weighting.DWA`
===========================

.. py:module:: LibMTL.weighting.DWA






.. py:class:: DWA

   Bases: :py:obj:`LibMTL.weighting.abstract_weighting.AbsWeighting`

   Dynamic Weight Average (DWA).

   This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

   :param T: The softmax temperature.
   :type T: float, default=2.0

   .. py:method:: backward(self, losses, **kwargs)




