:mod:`LibMTL.architecture.Cross_stitch`
=======================================

.. py:module:: LibMTL.architecture.Cross_stitch






.. py:class:: Cross_stitch(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)

   Bases: :py:obj:`LibMTL.architecture.abstract_arch.AbsArchitecture`

   Cross-stitch Networks (Cross_stitch).

   This method is proposed in `Cross-stitch Networks for Multi-task Learning (CVPR 2016) <https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf>`_ \
   and implemented by us.

   .. warning::
           - :class:`Cross_stitch` does not work with multiple inputs MTL problem, i.e., ``multi_input`` must be ``False``.

           - :class:`Cross_stitch` is only supported with ResNet-based encoder.




