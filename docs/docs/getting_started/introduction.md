## Introduction

``LibMTL`` is an open-source library built on [PyTorch](https://pytorch.org/) for Multi-Task Learning (MTL). This library has the following three characteristics.

- **Unified**:  ``LibMTL`` provides a unified code base to implement and a consistent evaluation procedure including data processing, metric objectives, and hyper-parameters on several representative MTL benchmark datasets, which allows quantitative, fair, and consistent comparisons between different MTL algorithms.
- **Comprehensive**: ``LibMTL`` supports many state-of-the-art MTL methods including 8 architectures and 16 optimization strategies. Meanwhile, ``LibMTL`` provides a fair comparison of several benchmark datasets covering different fields.
- **Extensible**:  ``LibMTL`` follows the modular design principles, which allows users to flexibly and conveniently add customized components or make personalized modifications. Therefore, users can easily and fast develop novel loss weighting strategies and architectures or apply the existing MTL algorithms to new application scenarios with the support of ``LibMTL``.



### Supported Algorithms

``LibMTL`` currently supports the following algorithms:

| Optimization Strategies                                                                                                                                                                                           | Venues             | Arguments                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ---------------------------------------------------------------------------------------------------------------- |
| Equal Weighting (EW)                                                                                                                                                                                              | -                  | ``--weighting EW``                                                                                                |
| Gradient Normalization ([GradNorm](http://proceedings.mlr.press/v80/chen18a/chen18a.pdf))                                                                                                                         | ICML 2018          | ``--weighting GradNorm``                                                                                                |
| Uncertainty Weights ([UW](https://openaccess.thecvf.com/content_cvpr_2018/papers/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.pdf))                                                                          | CVPR 2018          | ``--weighting UW``                                                                                               |
| [MGDA](https://papers.nips.cc/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html) ([official code](https://github.com/isl-org/MultiObjectiveOptimization))                                            | NeurIPS 2018       | ``--weighting MGDA``         |
| Dynamic Weight Average ([DWA](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) ([official code](https://github.com/lorenmt/mtan))   | CVPR 2019          | ``--weighting DWA``                               |
| Geometric Loss Strategy ([GLS](https://openaccess.thecvf.com/content_CVPRW_2019/papers/WAD/Chennupati_MultiNet_Multi-Stream_Feature_Aggregation_and_Geometric_Loss_Strategy_for_Multi-Task_CVPRW_2019_paper.pdf)) | CVPR 2019 Workshop | ``--weighting GLS``                                                                                                |
| Projecting Conflicting Gradient ([PCGrad](https://papers.nips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html))                                                                                 | NeurIPS 2020       | ``--weighting PCGrad``                                                                                                |
| Gradient sign Dropout ([GradDrop](https://papers.nips.cc/paper/2020/hash/16002f7a455a94aa4e91cc34ebdb9f2d-Abstract.html))                                                                                         | NeurIPS 2020       | ``--weighting GradDrop``                                                                                                |
| Impartial Multi-Task Learning ([IMTL](https://openreview.net/forum?id=IMPnRXEWpvr))                                                                                                                               | ICLR 2021          | ``--weighting IMTL``                                                                                                |
| Gradient Vaccine ([GradVac](https://openreview.net/forum?id=F1vEjWK-lH_))                                                                                                                                         | ICLR 2021          | ``--weighting GradVac``                                                                                                |
| Conflict-Averse Gradient descent ([CAGrad](https://openreview.net/forum?id=_61Qh8tULj_)) ([official code](https://github.com/Cranial-XIX/CAGrad))                                                                 | NeurIPS 2021       | ``--weighting CAGrad``                          |
| [Nash-MTL](https://proceedings.mlr.press/v162/navon22a/navon22a.pdf) ([official code](https://github.com/AvivNavon/nash-mtl))                                                                                     | ICML 2022          | ``--weighting Nash_MTL``                          |
| Random Loss Weighting ([RLW](https://openreview.net/forum?id=jjtFD8A1Wx))                                                                                                                                         | TMLR 2022          | ``--weighting RLW``                                                                                                |
| [MoCo](https://openreview.net/forum?id=dLAYGdKTi2)                                                                                                                                                                | ICLR 2023          | ``--weighting MoCo`` |
| [Aligned-MTL](https://openaccess.thecvf.com/content/CVPR2023/html/Senushkin_Independent_Component_Alignment_for_Multi-Task_Learning_CVPR_2023_paper.html) ([official code](https://github.com/SamsungLabs/MTL))   | CVPR 2023          | ``--weighting Aligned_MTL``                             |
| [DB-MTL](https://arxiv.org/abs/2308.12029)                                                                                                                                                                        | arXiv              | ``--weighting DB_MTL``                                                                                                |

| Architectures                                                                                                                                                           | Venues          | Arguments                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| Hard Parameter Sharing ([HPS](https://dl.acm.org/doi/10.5555/3091529.3091535))                                                                                          | ICML 1993       | ``--arch HPS``                                                                                                                  |
| Cross-stitch Networks ([Cross_stitch](https://openaccess.thecvf.com/content_cvpr_2016/papers/Misra_Cross-Stitch_Networks_for_CVPR_2016_paper.pdf))                      | CVPR 2016       | ``--arch Cross_stitch``                                                                                                                  |
| Multi-gate Mixture-of-Experts ([MMoE](https://dl.acm.org/doi/10.1145/3219819.3220007))                                                                                  | KDD 2018        | ``--arch MMoE``                                                                                                                  |
| Multi-Task Attention Network ([MTAN](https://openaccess.thecvf.com/content_CVPR_2019/papers/Liu_End-To-End_Multi-Task_Learning_With_Attention_CVPR_2019_paper.pdf)) ([official code](https://github.com/lorenmt/mtan))     | CVPR 2019       | ``--arch MTAN``                                                  |
| Customized Gate Control ([CGC](https://dl.acm.org/doi/10.1145/3383313.3412236)), Progressive Layered Extraction ([PLE](https://dl.acm.org/doi/10.1145/3383313.3412236)) | ACM RecSys 2020 |``--arch CGC``, ``--arch PLE``                                                                                                                  |
| Learning to Branch ([LTB](http://proceedings.mlr.press/v119/guo20e/guo20e.pdf))                                                                                         | ICML 2020       | ``--arch LTB``                                                                                                                  |
| [DSelect-k](https://proceedings.neurips.cc/paper/2021/hash/f5ac21cd0ef1b88e9848571aeb53551a-Abstract.html) ([official code](https://github.com/google-research/google-research/tree/master/dselect_k_moe))   | NeurIPS 2021    |``--arch DSelect_k``  |

### Supported Benchmark Datasets

| Datasets                                                                                 | Problems                      | Task Number  | Tasks                                                                      | multi-input | Supported Backbone             |
|:---------------------------------------------------------------------------------------- |:-----------------------------:|:------------:|:--------------------------------------------------------------------------:|:-----------:|:--------------------:|
| [NYUv2](https://github.com/median-research-group/LibMTL/tree/main/examples/nyu)          | Scene Understanding           | 3            | Semantic Segmentation+<br/>Depth Estimation+<br/>Surface Normal Prediction | ✘           | ResNet50/<br/>SegNet |
| [Office-31](https://github.com/median-research-group/LibMTL/tree/main/examples/office)   | Image Recognition             | 3            | Classification                                                             | ✓           | ResNet18             |
| [Office-Home](https://github.com/median-research-group/LibMTL/tree/main/examples/office) | Image Recognition             | 4            | Classification                                                             | ✓           | ResNet18             |
| [QM9](https://github.com/median-research-group/LibMTL/tree/main/examples/qm9)            | Molecular Property Prediction | 11 (default) | Regression                                                                 | ✘           | GNN                  |
| [PAWS-X](https://github.com/median-research-group/LibMTL/tree/main/examples/xtreme)      | Paraphrase Identification     | 4 (default)  | Classification                                                             | ✓           | Bert                 |

### Citation

If you find ``LibMTL`` useful for your research or development, please cite the following:

```latex
@article{lin2023libmtl,
  title={{LibMTL}: A {P}ython Library for Multi-Task Learning},
  author={Baijiong Lin and Yu Zhang},
  journal={Journal of Machine Learning Research},
  volume={24},
  number={209},
  pages={1--7},
  year={2023}
}
```

### Contributors

``LibMTL`` is developed and maintained by [Baijiong Lin](https://baijiong-lin.github.io).

### Contact Us

If you have any question or suggestion, please feel free to contact us by [raising an issue](https://github.com/median-research-group/LibMTL/issues) or sending an email to ``bj.lin.email@gmail.com``.

### Acknowledgements

We would like to thank the authors that release the public repositories (listed alphabetically):  [CAGrad](https://github.com/Cranial-XIX/CAGrad), [dselect_k_moe](https://github.com/google-research/google-research/tree/master/dselect_k_moe), [MultiObjectiveOptimization](https://github.com/isl-org/MultiObjectiveOptimization), [mtan](https://github.com/lorenmt/mtan), [MTL](https://github.com/SamsungLabs/MTL), [nash-mtl](https://github.com/AvivNavon/nash-mtl), [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric), and [xtreme](https://github.com/google-research/xtreme).
