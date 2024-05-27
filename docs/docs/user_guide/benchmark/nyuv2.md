## NYUv2

```eval_rst
The NYUv2 dataset :cite:`silberman2012indoor` is an indoor scene understanding dataset, which consists of video sequences recorded by the RGB and Depth cameras in the Microsoft Kinect. It contains 795 and 654 images with ground-truths for training and validation, respectively. 

We use the pre-processed NYUv2 dataset in :cite:`ljd19`, which can be downloaded `here <https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0>`_. Each input image has been resized to 3x288x384 and has labels for three tasks, including 13-class semantic segmentation, depth estimation, and surface normal prediction. Thus, it is a single-input problem, which means ``multi_input`` must be ``False``.

The training codes are mainly modified from `mtan <https://github.com/lorenmt/mtan>`_ and available in ``examples/nyu``. We use DeepLabV3+ architecture :cite:`ChenZPSA18`, where a ResNet-50 network pretrained on the ImageNet dataset with dilated convolutions :cite:`YuKF17` is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling (ASPP) module :cite:`ChenZPSA18` is used as task-specific head for each task. 

Following :cite:`ljd19`, the evaluation metrics of three tasks are adopted as follows. Mean Intersection over Union (mIoU) and Pixel Accuracy (Pix Acc) are used for the semantic segmentation task. Absolute and relative errors (denoted by Abs Err and Rel Err) are used for the depth estimation task. Five metrics are used for the surface normal estimation task: mean absolute of the error (Mean), median absolute of the error (Median), and percentage of pixels with the angular error below a threshold :math:`{\epsilon}` with :math:`{\epsilon}` as :math:`{11.25^{\circ}, 22.5^{\circ}, 30^{\circ}}` (abbreviated as <11.25, <22.5, <30), respectively. Among them, higher scores of mIoU, Pix Acc, <11.25, <22.5, and <30 mean better performance and lower scores of Abs Err, Rel Err, Mean, and Median indicate better performance.
```

### Run a Model

The script ``train_nyu.py`` is the main file for training and evaluating an MTL model on the NYUv2 dataset. A set of command-line arguments is provided to allow users to adjust the training configuration. 

Some important  arguments are described as follows.

```eval_rst
- ``weighting``: The weighting strategy. Refer to `here <../_autoapi/LibMTL/weighting/index.html>`_.
- ``arch``: The MTL architecture. Refer to `here <../_autoapi/LibMTL/architecture/index.html>`_.
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``scheduler``: The type of the learning rate scheduler. We recommend to use 'step' here.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``dataset_path``: The path of the NYUv2 dataset.
- ``aug``: If ``True``, the model is trained with a data augmentation.
- ``train_bs``: The batch size of training data. The default value is 8.
- ``test_bs``: The batch size of test data. The default value is 8.
```

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train an MTL model by executing the following command. 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH/nyuv2 --gpu_id GPU_ID --scheduler step
```

### References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```
