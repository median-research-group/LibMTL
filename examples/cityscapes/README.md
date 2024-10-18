## Cityscapes

The Cityscapes dataset [[1]](#1) is an urban scene understanding dataset. It contains 2975 and 500 images with ground-truths for training and validation, respectively. 

We use the pre-processed Cityscapes dataset in [[2]](#2), which can be downloaded [here](https://www.dropbox.com/sh/gaw6vh6qusoyms6/AADwWi0Tp3E3M4B2xzeGlsEna?dl=0). Each input image has been resized to <img title="" src="https://render.githubusercontent.com/render/math?math=3/times128/times256" alt=""> and has labels for two tasks, including 7-class semantic segmentation and depth estimation. Thus, it is a single-input problem, which means ``multi_input`` must be ``False``.

The training codes are mainly modified from [mtan](https://github.com/lorenmt/mtan). We use DeepLabV3+ architecture [[3]](#3), where a ResNet-50 network pretrained on the ImageNet dataset with dilated convolutions [[4]](#4) is used as a shared encoder among tasks and the Atrous Spatial Pyramid Pooling (ASPP) module [[3]](#3) is used as task-specific head for each task. 

Following [[2]](#2), the evaluation metrics of two tasks are adopted as follows. Mean Intersection over Union (mIoU) and Pixel Accuracy (Pix Acc) are used for the semantic segmentation task. Absolute and relative errors (denoted by Abs Err and Rel Err) are used for the depth estimation task. Among them, higher scores of mIoU and Pix Acc mean better performance and lower scores of Abs Err and Rel Err indicate better performance.

### Run a Model

The script ``main.py`` is the main file for training and evaluating an MTL model on the Cityscapes dataset. A set of command-line arguments is provided to allow users to adjust the training configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``scheduler``: The type of the learning rate scheduler. We recommend to use 'step' here.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``dataset_path``: The path of the Cityscapes dataset.
- ``train_bs``: The batch size of training data. The default value is 64.
- ``test_bs``: The batch size of test data. The default value is 64.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train an MTL model by executing the following command (This is based on the **DeepLabV3+** model.). 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH/cityscapes2 --gpu_id GPU_ID --scheduler step --mode train --save_path PATH
```

You can test the trained MTL model by running the following command.

```she
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH/cityscapes2 --gpu_id GPU_ID --scheduler step --mode test --load_path PATH
```

---

### References

<a id="1">[1]</a> Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The Cityscapes Dataset for Semantic Urban Scene Understanding. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2016.

<a id="2">[2]</a> Shikun Liu, Edward Johns, and Andrew J. Davison. End-to-End Multi-Task Learning with Attention. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2019.

<a id="3">[3]</a> Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation. In *European Conference on Computer Vision*, 2018.

<a id="4">[4]</a> Fisher Yu, Vladlen Koltun, and Thomas A. Funkhouser. Dilated Residual Networks. In *IEEE Conference on Computer Vision and Pattern Recognition*, 2017.
