:mod:`LibMTL.weighting.abstract_weighting`
==========================================

.. py:module:: LibMTL.weighting.abstract_weighting






.. py:class:: AbsWeighting

   Bases: :py:obj:`torch.nn.Module`

   An abstract class for weighting strategies.


   .. py:method:: init_param(self)

      Define and initialize some trainable parameters required by specific weighting methods.



   .. py:method:: backward(self, losses, **kwargs)
      :property:

      :param losses: A list of loss of each task.
      :type losses: list
      :param kwargs: A dictionary of hyperparameters of weighting methods.
      :type kwargs: dict




