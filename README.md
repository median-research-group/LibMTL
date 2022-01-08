# LibMTL
[![Documentation Status](https://readthedocs.org/projects/libmtl/badge/?version=latest)](https://libmtl.readthedocs.io/en/latest/?badge=latest) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/median-research-group/LibMTL/blob/main/LICENSE) [![PyPI version](https://badge.fury.io/py/LibMTL.svg)](https://badge.fury.io/py/LibMTL) [![Supported Python versions](https://img.shields.io/pypi/pyversions/LibMTL.svg?logo=python&logoColor=FFE873)](https://github.com/median-research-group/LibMTL) [![Downloads](https://static.pepy.tech/personalized-badge/libmtl?period=total&units=international_system&left_color=grey&right_color=red&left_text=downloads)](https://pepy.tech/project/libmtl) [![CodeFactor](https://www.codefactor.io/repository/github/median-research-group/libmtl/badge/main)](https://www.codefactor.io/repository/github/median-research-group/libmtl/overview/main) [![Maintainability](https://api.codeclimate.com/v1/badges/a1b5051e811eb6a1d837/maintainability)](https://codeclimate.com/github/median-research-group/LibMTL/maintainability) [![Made With Love](https://img.shields.io/badge/Made%20With-Love-orange.svg)](https://github.com/median-research-group/LibMTL) 

``LibMTL`` is an open-source library built on [PyTorch](https://pytorch.org/) for Multi-Task Learning (MTL). See the [latest documentation](https://libmtl.readthedocs.io/en/latest/) for more detailed introductions and API instructions.

## Features

- **Unified**:  ``LibMTL`` provides a unified code base to implement and a consistent evaluation procedure including data processing, metric objectives, and hyper-parameters on several representative MTL benchmark datasets, which allows quantitative, fair, and consistent comparisons between different MTL algorithms.
- **Comprehensive**: ``LibMTL`` supports 84 MTL models combined by 7 architectures and 12 loss weighting strategies. Meanwhile, ``LibMTL`` provides a fair comparison on 3 computer vision datasets.
- **Extensible**:  ``LibMTL`` follows the modular design principles, which allows users to flexibly and conveniently add customized components or make personalized modifications. Therefore, users can easily and fast develop novel loss weighting strategies and architectures or apply the existing MTL algorithms to new application scenarios with the support of ``LibMTL``.

## Overall Framework

 ![framework](./docs/docs/images/framework.png).

- **Config Module**: Responsible for all the configuration parameters involved in the running framework, including the parameters of optimizer and learning rate scheduler, the hyper-parameters of MTL model, training configuration like batch size, total epoch, random seed and so on.
- **Dataloaders Module**: Responsible for data pre-processing and loading.
- **Model Module**: Responsible for inheriting classes architecture and weighting and instantiating a MTL model. Note that the architecture and the weighting strategy determine the forward and backward processes of the MTL model, respectively.
- **Losses Module**: Responsible for computing the loss for each task. 
- **Metrics Module**: Responsible for evaluating the MTL model and calculating the metric scores for each task.

## Supported Algorithms

``LibMTL`` currently supports the following algorithms:

- 12 loss weighting strategies.

| Weighting Strategy                                           | Venues              | Comments                                                     |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| Equally Weighting (EW)                                       | -                   | Implemented by us                                            |
| Gradient Normalization ([GradNorm](http://proceedings.mlr.press/v80/chen18a/chen18a.pdf)) | ICML 2018           | Implemented by us                                            |
| Uncertainty Weights ([UW](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)) | CVPR 2018           | Implemented by us                                            |
| [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) | NeurIPS 2018        | Referenced from [official PyTorch implementation](https://github.com/isl-org/MultiObjectiveOptimization) |
| Dynamic Weight Average ([DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) | CVPR 2019           | Referenced from [official PyTorch implementation](https://github.com/lorenmt/mtan) |
| Geometric Loss Strategy ([GLS](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)) | CVPR 2019 workshop  | Implemented by us                                            |
| Projecting Conflicting Gradient ([PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)) | NeurIPS 2020        | Implemented by us                                            |
| Gradient sign Dropout ([GradDrop](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html)) | NeurIPS 2020        | Implemented by us                                            |
| Impartial Multi-Task Learning ([IMTL](https://openreview.net/forum?id=IMPnRXEWpvr)) | ICLR 2021           | Implemented by us                                            |
| Gradient Vaccine ([GradVac](https://openreview.net/forum?id=F1vEjWK-lH_)) | ICLR 2021 Spotlight | Implemented by us                                            |
| Conflict-Averse Gradient descent ([CAGrad](https://openreview.net/forum?id=_61Qh8tULj_)) | NeurIPS 2021        | Referenced from [official PyTorch implementation](https://github.com/Cranial-XIX/CAGrad) |
| Random Loss Weighting ([RLW](https://arxiv.org/abs/2111.10603)) | arXiv               | Implemented by us                                            |

- 7 architectures.

| Architecture                                                 | Venues                     | Comments                                                     |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| Hrad Parameter Sharing ([HPS](https://dl.acm.org/doi/10.5555/3091529.3091535)) | ICML 1993                  | Implemented by us                                            |
| Cross-stitch Networks ([Cross_stitch](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf)) | CVPR 2016                  | Implemented by us                                            |
| Multi-gate Mixture-of-Experts ([MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007)) | KDD 2018                   | Implemented by us                                            |
| Multi-Task Attention Network ([MTAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) | CVPR 2019                  | Referenced from [official PyTorch implementation](https://github.com/lorenmt/mtan) |
| Customized Gate Control ([CGC](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 Best Paper | Implemented by us                                            |
| Progressive Layered Extraction ([PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 Best Paper | Implemented by us                                            |
| [DSelect-k](https://openreview.net/forum?id=tKlYQJLYN8v)     | NeurIPS 2021               | Referenced from [official TensorFlow implementation](https://github.com/google-research/google-research/tree/master/dselect_k_moe) |

- 84 combinations of different architectures and loss weighting strategies.

## Installation

The simplest way to install `LibMTL` is using `pip`.

```shell
pip install -U LibMTL
```

More details about environment configuration is represented in [Docs](https://libmtl.readthedocs.io/en/latest/docs/getting_started/installation.html).

## Quick Start

We use the NYUv2 dataset as an example to show how to use ``LibMTL``.

### Download Dataset

The NYUv2 dataset we used is pre-processed by [mtan](https://github.com/lorenmt/mtan). You can download this dataset [here](https://www.dropbox.com/sh/86nssgwm6hm3vkb/AACrnUQ4GxpdrBbLjb6n-mWNa?dl=0).

### Run a Model

The complete training code for the NYUv2 dataset is provided in [examples/nyu](./examples/nyu). The file [train_nyu.py](./examples/nyu/train_nyu.py) is the main file for training on the NYUv2 dataset.

You can find the command-line arguments by running the following command.

```shell
python train_nyu.py -h
```

For instance, running the following command will train a MTL model with EW and HPS on NYUv2 dataset.

```shell
python train_nyu.py --weighting EW --arch HPS --dataset_path /path/to/nyuv2 --gpu_id 0 --scheduler step
```

More details is represented in [Docs](https://libmtl.readthedocs.io/en/latest/docs/getting_started/quick_start.html).

## Citation

If you find ``LibMTL`` useful for your research or development, please cite the following:

```latex
@misc{LibMTL,
 author = {Baijiong Lin and Yu Zhang},
 title = {LibMTL: A PyTorch Library for Multi-Task Learning},
 year = {2021},
 publisher = {GitHub},
 journal = {GitHub repository},
 howpublished = {\url{https://github.com/median-research-group/LibMTL}}
}
```

## Contributors

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io) and [Yu Zhang](http://cse.sustech.edu.cn/faculty/~zhangy/).

### Contact Us

If you have any question or suggestion, please feel free to contact us by [raising an issue](https://github.com/median-research-group/LibMTL/issues) or sending an email to ``linbj@mail.sustech.edu.cn``.

## Acknowledgements

We would like to thank the authors that release the public repositories (listed alphabetically):  [CAGrad](https://github.com/Cranial-XIX/CAGrad), [dselect_k_moe](https://github.com/google-research/google-research/tree/master/dselect_k_moe), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), and [mtan](https://github.com/lorenmt/mtan).
