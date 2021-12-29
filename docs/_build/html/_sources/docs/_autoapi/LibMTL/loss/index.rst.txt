:mod:`LibMTL.loss`
==================

.. py:module:: LibMTL.loss






.. py:class:: AbsLoss

   Bases: :py:obj:`object`

   An abstract class for loss function.


   .. py:method:: compute_loss(self, pred, gt)
      :property:

      Calculate the loss.

      :param pred: The prediction tensor.
      :type pred: torch.Tensor
      :param gt: The ground-truth tensor.
      :type gt: torch.Tensor

      :returns: The loss.
      :rtype: torch.Tensor



.. py:class:: CELoss

   Bases: :py:obj:`AbsLoss`

   The cross entropy loss function.


   .. py:method:: compute_loss(self, pred, gt)

      




