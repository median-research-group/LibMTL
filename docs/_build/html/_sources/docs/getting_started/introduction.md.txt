## Introduction

``LibMTL`` is a open-source library built with [PyTorch](https://pytorch.org/) for Multi-Task Learning (MTL) research and applications. This library has three characteristics as follows.

- **Unified**:  ``LibMTL`` provides a unified code base to implement and a consistent evaluation procedure including data processing, metric objectives, and hyper-parameters on several representative MTL benchmark datasets, which allows quantitative, fair, and consistent comparisons between different algorithms.
- **Comprehensive**: ``LibMTL`` supports 84 MTL algorithms combined by 7 architectures and 12 weighting strategies. Meanwhile, ``LibMTL`` provides a fair comparison on 3 computer vision datasets.
- **Extensible**:  The construction of ``LibMTL`` follows the modular design principles, which allows users to flexibly and conveniently add customized components or make personalized modifications. Therefore, researchers and applicators can easily and fast develop novel MTL weighting strategies and architectures or apply the existing MTL algorithms to new application scenarios with the support of ``LibMTL``.



### Supported Algorithms

``LibMTL`` currently supports the following algorithms:

- 12 weighting strategies.

| Weighting Strategy                                           | Venues              | Comments                                                     |
| ------------------------------------------------------------ | ------------------- | ------------------------------------------------------------ |
| Equally Weighting (EW)                                       | -                   | Implemented by us                                            |
| Gradient Normalization ([GradNorm](http://proceedings.mlr.press/v80/chen18a/chen18a.pdf)) | ICML 2018           | Implemented by us                                            |
| Uncertainty Weights ([UW](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf)) | CVPR 2018           | Implemented by us                                            |
| [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) | NeurIPS 2018        | Modified from [official PyTorch implementation](https://github.com/isl-org/MultiObjectiveOptimization) |
| Dynamic Weight Average ([DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) | CVPR 2019           | Modified from [official PyTorch implementation](https://github.com/lorenmt/mtan) |
| Geometric Loss Strategy ([GLS](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)) | CVPR 2019 workshop  | Implemented by us                                            |
| Projecting Conflicting Gradient ([PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)) | NeurIPS 2020        | Implemented by us                                            |
| Gradient sign Dropout ([GradDrop](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html)) | NeurIPS 2020        | Implemented by us                                            |
| Impartial Multi-Task Learning ([IMTL](https://openreview.net/forum?id=IMPnRXEWpvr)) | ICLR 2021           | Implemented by us                                            |
| Gradient Vaccine ([GradVac](https://openreview.net/forum?id=F1vEjWK-lH_)) | ICLR 2021 Spotlight | Implemented by us                                            |
| Conflict-Averse Gradient descent ([CAGrad](https://openreview.net/forum?id=_61Qh8tULj_)) | NeurIPS 2021        | Modified from [official PyTorch implementation](https://github.com/Cranial-XIX/CAGrad) |
| Random Loss Weighting ([RLW](https://arxiv.org/abs/2111.10603)) | arXiv               | Implemented by us                                            |

- 7 architectures.

| Architecture                                                 | Venues                     | Comments                                                     |
| ------------------------------------------------------------ | -------------------------- | ------------------------------------------------------------ |
| Hrad Parameter Sharing ([HPS](https://dl.acm.org/doi/10.5555/3091529.3091535)) | ICML 1993                  | Implemented by us                                            |
| Cross-stitch Networks ([Cross_stitch](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf)) | CVPR 2016                  | Implemented by us                                            |
| Multi-gate Mixture-of-Experts ([MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007)) | KDD 2018                   | Implemented by us                                            |
| Multi-Task Attention Network ([MTAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) | CVPR 2019                  | Modified from [official PyTorch implementation](https://github.com/lorenmt/mtan) |
| Customized Gate Control ([CGC](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 Best Paper | Implemented by us                                            |
| Progressive Layered Extraction ([PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 Best Paper | Implemented by us                                            |
| [DSelect-k](https://openreview.net/forum?id=tKlYQJLYN8v)     | NeurIPS 2021               | Modified from [official TensorFlow implementation](https://github.com/google-research/google-research/tree/master/dselect_k_moe) |

- 84 combinations of different architectures and weighting strategies.

### Citation

If you find ``LibMTL`` useful for your research or development, please citing the following:

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

### Contributors

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io) and [Yu Zhang](http://cse.sustech.edu.cn/faculty/~zhangy/).

### Contact Us

If you have any question or suggestion, please feel free to contact us by [raising an issue](https://github.com/median-research-group/LibMTL/issues) or sending an email to ``linbj@mail.sustech.edu.cn``.

### Acknowledgements

We would like to thank the authors that release the public repositories as follows (listed alphabetically):  [CAGrad](https://github.com/Cranial-XIX/CAGrad), [dselect_k_moe](https://github.com/google-research/google-research/tree/master/dselect_k_moe), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), and [mtan](https://github.com/lorenmt/mtan).