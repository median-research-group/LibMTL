## NYUv2

The NYUv2 dataset [[1]](#1) is an indoor scene understanding dataset, which consists of video sequences recorded by the RGB and Depth cameras in the Microsoft Kinect. It contains 795 and 654 images with ground-truths for training and validation, respectively. 

We use the pre-processed NYUv2 dataset in [[2]](#2), which can be downloaded [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0). Each input image has been resized to <img src="https://render.githubusercontent.com/render/math?math=3\times288\times384"> and has labels for three tasks, including 13-class semantic segmentation, depth estimation, and surface normal prediction. Thus, it is a single-input problem, which means ``multi_input`` must be ``False``.

The training codes are mainly modified from [mtan](https://github.com/lorenmt/mtan). We use DeepLabV3+ architecture [[3]](#3), where a ResNet-50 network pretrained on the ImageNet dataset with dilated convolutions [[4]](#4) is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling (ASPP) module [[3]](#3) is used as task-specific head for each task. 

Following [[2]](#2), the evaluation metrics of three tasks are adopted as follows. Mean Intersection over Union (mIoU) and Pixel Accuracy (Pix Acc) are used for the semantic segmentation task. Absolute and relative errors (denoted by Abs Err and Rel Err) are used for the depth estimation task. Five metrics are used for the surface normal estimation task: mean absolute of the error (Mean), median absolute of the error (Median), and percentage of pixels with the angular error below a threshold <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> with <img src="https://render.githubusercontent.com/render/math?math=\epsilon"> as <img src="https://render.githubusercontent.com/render/math?math=11.25^{\circ}, 22.5^{\circ}, 30^{\circ}"> (abbreviated as <11.25, <22.5, <30), respectively. Among them, higher scores of mIoU, Pix Acc, <11.25, <22.5, and <30 mean better performance and lower scores of Abs Err, Rel Err, Mean, and Median indicate better performance.

### Run a Model

The script ``main.py`` is the main file for training and evaluating an MTL model on the NYUv2 dataset. A set of command-line arguments is provided to allow users to adjust the training configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``scheduler``: The type of the learning rate scheduler. We recommend to use 'step' here.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``dataset_path``: The path of the NYUv2 dataset.
- ``aug``: If ``True``, the model is trained with a data augmentation.
- ``train_bs``: The batch size of training data. The default value is 8.
- ``test_bs``: The batch size of test data. The default value is 8.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train an MTL model by executing the following command (This is based on the **DeepLabV3+** model.). 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH/nyuv2 --gpu_id GPU_ID --scheduler step --mode train --save_path PATH
```

You can test the trained MTL model by running the following command.

```she
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH/nyuv2 --gpu_id GPU_ID --scheduler step --mode test --load_path PATH
```

---

We also provide the **SegNet+MTAN** model, which is used in [[2]](#2), [[5]](#5) and [[6]](#6). You can train it by executing the following command. 

```shell
python main_segnet.py --weighting WEIGHTING --dataset_path PATH/nyuv2 --gpu_id GPU_ID --scheduler step --aug --weight_decay 0 --mode train --save_path PATH
```

### References

<a id="1">[1]</a> Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor Segmentation and Support Inference from RGBD Images. In *European Conference on Computer Vision*, 2012.

<a id="2">[2]</a> Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-End Multi-Task Learning with Attention. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

<a id="3">[3]</a> Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In *European Conference on Computer Vision*, 2018.

<a id="4">[4]</a> Fisher Yu, Vladlen Koltun, and Thomas A. Funkhouser. Dilated Residual Networks. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2017.

<a id="5">[5]</a> Bo Liu, Xingchao Liu, Xiaojie Jin, Peter Stone, and Qiang Liu. Conflict-Averse Gradient Descent for Multi-Task Learning. In *Neural Information Processing Systems*, 2021.

<a id="6">[6]</a> Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-Task Learning as A Bargaining Game. In *International Conference on Machine Learning*, 2022.
