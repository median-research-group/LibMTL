:mod:`LibMTL.utils`
===================

.. py:module:: LibMTL.utils






.. py:function:: set_random_seed(seed)

   Set the random seed for reproducibility.

   :param seed: The random seed.
   :type seed: int, default=0


.. py:function:: set_device(gpu_id)

   Set the device where model and data will be allocated.

   :param gpu_id: The id of gpu.
   :type gpu_id: str, default='0'


.. py:function:: count_parameters(model)

   Calculates the number of parameters for a model.

   :param model: A neural network module.
   :type model: torch.nn.Module


.. py:function:: count_improvement(base_result, new_result, weight)

   Calculate the improvement between two results,

   .. math::
       \Delta_{\mathrm{p}}=100\%\times \frac{1}{T}\sum_{t=1}^T
       \frac{1}{M_t}\sum_{m=1}^{M_t}\frac{(-1)^{w_{t,m}}(B_{t,m}-N_{t,m})}{N_{t,m}}.

   :param base_result: A dictionary of scores of all metrics of all tasks.
   :type base_result: dict
   :param new_result: The same structure with ``base_result``.
   :type new_result: dict
   :param weight: The same structure with ``base_result`` while each elements is binary integer representing whether higher or lower score is better.
   :type weight: dict

   :returns: The improvement between ``new_result`` and ``base_result``.
   :rtype: float

   Examples::

       base_result = {'A': [96, 98], 'B': [0.2]}
       new_result = {'A': [93, 99], 'B': [0.5]}
       weight = {'A': [1, 0], 'B': [1]}

       print(count_improvement(base_result, new_result, weight))



