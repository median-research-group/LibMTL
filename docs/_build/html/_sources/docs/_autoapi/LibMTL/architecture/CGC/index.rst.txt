:mod:`LibMTL.architecture.CGC`
==============================

.. py:module:: LibMTL.architecture.CGC






.. py:class:: CGC(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.MMoE.MMoE`

   Customized Gate Control (CGC).

   This method is proposed in `Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (ACM RecSys 2020 Best Paper) <https://dl.acm.org/doi/10.1145/3383313.3412236>`_ \
   and implemented by us.

   :param img_size: The size of input data. For example, [3, 244, 244] for input images with size 3x224x224.
   :type img_size: list
   :param num_experts: The numbers of experts shared for all tasks and specific to each task, respectively. Each expert is the encoder network.
   :type num_experts: list

   .. py:method:: forward(self, inputs, task_name=None)




