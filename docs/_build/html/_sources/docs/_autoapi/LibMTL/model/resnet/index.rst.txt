:mod:`LibMTL.model.resnet`
==========================

.. py:module:: LibMTL.model.resnet






.. py:data:: model_urls
   

   

.. py:function:: conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1)

   3x3 convolution with padding


.. py:function:: conv1x1(in_planes, out_planes, stride=1)

   1x1 convolution


.. py:class:: BasicBlock(inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None)

   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super(Model, self).__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. py:attribute:: expansion
      :annotation: = 1

      

   .. py:method:: forward(self, x)

      Defines the computation performed at every call.

      Should be overridden by all subclasses.

      .. note::
          Although the recipe for forward pass needs to be defined within
          this function, one should call the :class:`Module` instance afterwards
          instead of this since the former takes care of running the
          registered hooks while the latter silently ignores them.



.. py:class:: Bottleneck(inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=None)

   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super(Model, self).__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. py:attribute:: expansion
      :annotation: = 4

      

   .. py:method:: forward(self, x)

      Defines the computation performed at every call.

      Should be overridden by all subclasses.

      .. note::
          Although the recipe for forward pass needs to be defined within
          this function, one should call the :class:`Module` instance afterwards
          instead of this since the former takes care of running the
          registered hooks while the latter silently ignores them.



.. py:class:: ResNet(block, layers, num_classes=1000, zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None, norm_layer=None)

   Bases: :py:obj:`torch.nn.Module`

   Base class for all neural network modules.

   Your models should also subclass this class.

   Modules can also contain other Modules, allowing to nest them in
   a tree structure. You can assign the submodules as regular attributes::

       import torch.nn as nn
       import torch.nn.functional as F

       class Model(nn.Module):
           def __init__(self):
               super(Model, self).__init__()
               self.conv1 = nn.Conv2d(1, 20, 5)
               self.conv2 = nn.Conv2d(20, 20, 5)

           def forward(self, x):
               x = F.relu(self.conv1(x))
               return F.relu(self.conv2(x))

   Submodules assigned in this way will be registered, and will have their
   parameters converted too when you call :meth:`to`, etc.

   .. py:method:: forward(self, x)

      Defines the computation performed at every call.

      Should be overridden by all subclasses.

      .. note::
          Although the recipe for forward pass needs to be defined within
          this function, one should call the :class:`Module` instance afterwards
          instead of this since the former takes care of running the
          registered hooks while the latter silently ignores them.



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



