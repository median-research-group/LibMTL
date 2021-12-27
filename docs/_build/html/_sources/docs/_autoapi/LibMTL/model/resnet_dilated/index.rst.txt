:mod:`LibMTL.model.resnet_dilated`
==================================

.. py:module:: LibMTL.model.resnet_dilated






.. py:class:: ResnetDilated(orig_resnet, dilate_scale=8)

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


   .. py:method:: forward_stage(self, x, stage)



.. py:function:: resnet_dilated(basenet, pretrained=True, dilate_scale=8)

   Dilated Residual Network models from `"Dilated Residual Networks" <https://openaccess.thecvf.com/content_cvpr_2017/papers/Yu_Dilated_Residual_Networks_CVPR_2017_paper.pdf>`_

   :param basenet: The type of ResNet.
   :type basenet: str
   :param pretrained: If True, returns a model pre-trained on ImageNet.
   :type pretrained: bool
   :param dilate_scale: The type of dilating process.
   :type dilate_scale: {8, 16}, default=8



