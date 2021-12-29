:mod:`LibMTL.architecture.DSelect_k`
====================================

.. py:module:: LibMTL.architecture.DSelect_k






.. py:class:: DSelect_k(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.MMoE.MMoE`

   DSelect-k.

   This method is proposed in `DSelect-k: Differentiable Selection in the Mixture of Experts with Applications to Multi-Task Learning (NeurIPS 2021) <https://openreview.net/forum?id=tKlYQJLYN8v>`_ \
   and implemented by modifying from the `official TensorFlow implementation <https://github.com/google-research/google-research/tree/master/dselect_k_moe>`_.

   :param img_size: The size of input data. For example, [3, 244, 244] for input images with size 3x224x224.
   :type img_size: list
   :param num_experts: The number of experts shared for all tasks. Each expert is the encoder network.
   :type num_experts: int
   :param num_nonzeros: The number of selected experts.
   :type num_nonzeros: int
   :param kgamma: A scaling parameter for the smooth-step function.
   :type kgamma: float, default=1.0

   .. py:method:: forward(self, inputs, task_name=None)




