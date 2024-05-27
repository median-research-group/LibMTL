## Quick Start

```eval_rst
We use the NYUv2 dataset :cite:`silberman2012indoor` as an example to show how to use ``LibMTL``. More details and results are provided here.
```

### Download Dataset

The NYUv2 dataset we used is pre-processed by [mtan](https://github.com/lorenmt/mtan). You can download this dataset [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0). The directory structure is as follows:

```shell
*/nyuv2/
├── train
│   ├── depth
│   ├── image
│   ├── label
│   └── normal
└── val
    ├── depth
    ├── image
    ├── label
    └── normal
```

The NYUv2 dataset is a MTL benchmark dataset, which includes three tasks: 13-class semantic segmentation, depth estimation, and surface normal prediction. ``image`` contains the input images and ``label``, ``depth``, ``normal`` contains the labels for three tasks, respectively. We train the MTL model with the data in ``train`` and evaluate on ``val``. 

### Run a Model

The complete training code for the NYUv2 dataset is provided in [examples/nyu](https://github.com/median-research-group/LibMTL/examples/nyu). The file ``train_nyu.py`` is the main file for training on the NYUv2 dataset.

You can find the command-line arguments by running the following command.

```shell
python main.py -h
```

```eval_rst
For instance, running the following command will train a MTL model with :class:`LibMTL.weighting.EW` and :class:`LibMTL.architecture.HPS` on NYUv2 dataset.
```

```shell
python main.py --weighting EW --arch HPS --dataset_path /path/to/nyuv2 --gpu_id 0 --scheduler step
```

If everything works fine, you will see the following outputs which includes the training configurations and the number of model parameters.

```
========================================
General Configuration:
    Wighting: EW
    Architecture: HPS
    Rep_Grad: False
    Multi_Input: False
    Seed: 0
    Device: cuda:0
Optimizer Configuration:
    optim: adam
    lr: 0.0001
    weight_decay: 1e-05
Scheduler Configuration:
    scheduler: step
    step_size: 100
    gamma: 0.5
========================================
Total Params: 71888721
Trainable Params: 71888721
Non-trainable Params: 0
========================================
```

Next, the results will be printed in following format.

```
LOG FORMAT | segmentation_LOSS mIoU pixAcc | depth_LOSS abs_err rel_err | normal_LOSS mean median <11.25 <22.5 <30 | TIME
Epoch: 0000 | TRAIN: 1.4417 0.2494 0.5717 | 1.4941 1.4941 0.5002 | 0.3383 43.1593 38.2601 0.0913 0.2639 0.3793 | Time: 81.6612 | TEST: 1.0898 0.3589 0.6676 | 0.7027 0.7027 0.2615 | 0.2143 32.8732 29.4323 0.1734 0.3878 0.5090 | Time: 11.9699
Epoch: 0001 | TRAIN: 0.8958 0.4194 0.7201 | 0.7011 0.7011 0.2448 | 0.1993 31.5235 27.8404 0.1826 0.4060 0.5361 | Time: 82.2399 | TEST: 0.9980 0.4189 0.6868 | 0.6274 0.6274 0.2347 | 0.1991 31.0144 26.5077 0.2065 0.4332 0.5551 | Time: 12.0278
```

If the training process ends, the best result on ``val`` will be printed as follows.

```
Best Result: Epoch 65, result {'segmentation': [0.5377492904663086, 0.7544658184051514], 'depth': [0.38453552363844823, 0.1605487049810748], 'normal': [23.573742, 17.04381, 0.35038458555943763, 0.609274380451927, 0.7207172795833373]}
```

### References

```eval_rst
.. bibliography::
   :style: unsrt
   :filter: docname in docnames
```
