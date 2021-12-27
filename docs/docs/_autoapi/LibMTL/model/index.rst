:mod:`LibMTL.model`
===================

.. py:module:: LibMTL.model






.. py:function:: resnet18(pretrained=False, progress=True, **kwargs)

   ResNet-18 model from
   `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnet34(pretrained=False, progress=True, **kwargs)

   ResNet-34 model from
   `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnet50(pretrained=False, progress=True, **kwargs)

   ResNet-50 model from
   `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnet101(pretrained=False, progress=True, **kwargs)

   ResNet-101 model from
   `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnet152(pretrained=False, progress=True, **kwargs)

   ResNet-152 model from
   `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnext50_32x4d(pretrained=False, progress=True, **kwargs)

   ResNeXt-50 32x4d model from
   `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnext101_32x8d(pretrained=False, progress=True, **kwargs)

   ResNeXt-101 32x8d model from
   `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: wide_resnet50_2(pretrained=False, progress=True, **kwargs)

   Wide ResNet-50-2 model from
   `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

   The model is the same as ResNet except for the bottleneck number of channels
   which is twice larger in every block. The number of channels in outer 1x1
   convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
   channels, and in Wide ResNet-50-2 has 2048-1024-2048.

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: wide_resnet101_2(pretrained=False, progress=True, **kwargs)

   Wide ResNet-101-2 model from
   `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

   The model is the same as ResNet except for the bottleneck number of channels
   which is twice larger in every block. The number of channels in outer 1x1
   convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
   channels, and in Wide ResNet-50-2 has 2048-1024-2048.

   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param progress: If True, displays a progress bar of the download to stderr.
   :type progress: bool


.. py:function:: resnet_dilated(basenet, pretrained=True, dilate_scale=8)

   Dilated Residual Network models from `"Dilated Residual Networks" <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf>`_

   :param basenet: The type of ResNet.
   :type basenet: str
   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param dilate_scale: The type of dilating process.
   :type dilate_scale: {8, 16}, default=8



