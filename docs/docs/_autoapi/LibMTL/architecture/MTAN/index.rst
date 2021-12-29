:mod:`LibMTL.architecture.MTAN`
===============================

.. py:module:: LibMTL.architecture.MTAN






.. py:class:: MTAN(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Multi-Task Attention Network (MTAN).

   This method is proposed in `End-To-End Multi-Task Learning With Attention (CVPR 2019) <https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf>`_ \
   and implemented by modifying from the `official PyTorch implementation <https://github.com/lorenmt/mtan>`_.

   .. warning::
           :class:`MTAN` is only supported with ResNet-based encoder.


   .. py:method:: forward(self, inputs, task_name=None)


   .. py:method:: get_share_params(self)


   .. py:method:: zero_grad_share_params(self)




