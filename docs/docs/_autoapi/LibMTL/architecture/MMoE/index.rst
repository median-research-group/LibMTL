:mod:`LibMTL.architecture.MMoE`
===============================

.. py:module:: LibMTL.architecture.MMoE






.. py:class:: MMoE(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Multi-gate Mixture-of-Experts (MMoE).

   This method is proposed in `Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts (KDD 2018) <https://dl.acm.org/doi/10.1145/3219819.3220007>`_ \
   and implemented by us.

   :param img_size: The size of input data. For example, [3, 244, 244] for input images with size 3x224x224.
   :type img_size: list
   :param num_experts: The number of experts shared for all tasks. Each expert is the encoder network.
   :type num_experts: int

   .. py:method:: forward(self, inputs, task_name=None)


   .. py:method:: get_share_params(self)


   .. py:method:: zero_grad_share_params(self)




