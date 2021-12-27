:mod:`LibMTL.metrics`
=====================

.. py:module:: LibMTL.metrics






.. py:class:: AbsMetric

   Bases: :py:obj:`object`

   An abstract class for the performance metrics of a task.

   .. attribute:: record

      A list of the metric scores in every iteration.

      :type: list

   .. attribute:: bs

      A list of the number of data in every iteration.

      :type: list

   .. py:method:: update_fun(self, pred, gt)
      :property:

      Calculate the metric scores in every iteration and update :attr:`record`.

      :param pred: The prediction tensor.
      :type pred: torch.Tensor
      :param gt: The ground-truth tensor.
      :type gt: torch.Tensor


   .. py:method:: score_fun(self)
      :property:

      Calculate the final score (when a epoch ends).

      :returns: A list of metric scores.
      :rtype: list


   .. py:method:: reinit(self)

      Reset :attr:`record` and :attr:`bs` (when a epoch ends).




.. py:class:: AccMetric

   Bases: :py:obj:`AbsMetric`

   Calculate the accuracy.


   .. py:method:: update_fun(self, pred, gt)

      


   .. py:method:: score_fun(self)

      




