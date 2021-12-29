:mod:`LibMTL.architecture.abstract_arch`
========================================

.. py:module:: LibMTL.architecture.abstract_arch






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





