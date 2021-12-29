:mod:`LibMTL.architecture`
==========================

.. py:module:: LibMTL.architecture






.. py:class:: AbsArchitecture(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`torch.nn.Module`

   An abstract class for MTL architectures.

   :param task_name: A list of strings for all tasks.
   :type task_name: list
   :param encoder_class: A neural network class.
   :type encoder_class: class
   :param decoders: A dictionary of name-decoder pairs of type (:class:`str`, :class:`torch.nn.Module`).
   :type decoders: dict
   :param rep_grad: If ``True``, the gradient of the representation for each task can be computed.
   :type rep_grad: bool
   :param multi_input: Is ``True`` if each task has its own input data, ``False`` otherwise.
   :type multi_input: bool
   :param device: The device where model and data will be allocated.
   :type device: torch.device
   :param kwargs: A dictionary of hyperparameters of architecture methods.
   :type kwargs: dict

   .. py:method:: forward(self, inputs, task_name=None)

      :param inputs: The input data.
      :type inputs: torch.Tensor
      :param task_name: The task name corresponding to ``inputs`` if ``multi_input`` is ``True``.
      :type task_name: str, default=None

      :returns: A dictionary of name-prediction pairs of type (:class:`str`, :class:`torch.Tensor`).
      :rtype: dict


   .. py:method:: get_share_params(self)

      Return the shared parameters of the model.



   .. py:method:: zero_grad_share_params(self)

      Set gradients of the shared parameters to zero.




.. py:class:: HPS(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Hrad Parameter Sharing (HPS).

   This method is proposed in `Multitask Learning: A Knowledge-Based Source of Inductive Bias (ICML 1993) <https://dl.acm.org/doi/10.5555/3091529.3091535>`_ \
   and implemented by us.


.. py:class:: Cross_stitch(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Cross-stitch Networks (Cross_stitch).

   This method is proposed in `Cross-stitch Networks for Multi-task Learning (CVPR 2016) <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>`_ \
   and implemented by us.

   .. warning::
           - :class:`Cross_stitch` does not work with multiple inputs MTL problem, i.e., ``multi_input`` must be ``False``.

           - :class:`Cross_stitch` is only supported with ResNet-based encoder.



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




