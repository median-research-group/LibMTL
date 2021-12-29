:mod:`LibMTL.architecture.PLE`
==============================

.. py:module:: LibMTL.architecture.PLE






.. py:class:: PLE(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Progressive Layered Extraction (PLE).

   This method is proposed in `Progressive Layered Extraction (PLE): A Novel Multi-Task Learning (MTL) Model for Personalized Recommendations (ACM RecSys 2020 Best Paper) <https://dl.acm.org/doi/10.1145/3383313.3412236>`_ \
   and implemented by us.

   :param img_size: The size of input data. For example, [3, 244, 244] for input images with size 3x224x224.
   :type img_size: list
   :param num_experts: The numbers of experts shared for all tasks and specific to each task, respectively. Each expert is the encoder network.
   :type num_experts: list

   .. warning::
           - :class:`PLE` does not work with multiple inputs MTL problem, i.e., ``multi_input`` must be ``False``.
           - :class:`PLE` is only supported with ResNet-based encoder.


   .. py:method:: forward(self, inputs, task_name=None)


   .. py:method:: get_share_params(self)


   .. py:method:: zero_grad_share_params(self)




