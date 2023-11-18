## QM9

The QM9 dataset [[1]](#1) consists of about 130K molecules with 19 regression targets. The training codes are mainly followed [[2]](#2) and modified from [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/qm9_nn_conv.py). 

### Install PyG

```shell
pip install torch-scatter==2.0.8 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch_sparse==0.6.10 -f https://pytorch-geometric.com/whl/torch-1.8.1+cu111.html
pip install torch_geometric==2.2.0
```

### Run a Model

The script ``main.py`` is the main file for training and evaluating a MTL model on the QM9 dataset. A set of command-line arguments is provided to allow users to adjust the training parameter configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``optim``: The type of the optimizer. We recommend to use 'adam' here.
- ``target``: The index of target tasks.
- ``dataset_path``: The path of the QM9 dataset.
- ``bs``: The batch size of training, validation, and test data. The default value is 128.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train a MTL model by running a command like this. 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --target TARGET --mode train --save_path PATH
```

You can test the trained MTL model by running the following command.

```she
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --target TARGET --mode test --load_path PATH
```

### References

<a id="1">[1]</a> Zhenqin Wu, Bharath Ramsundar, Evan N. Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S. Pappu, Karl Leswing, and Vijay Pande. MoleculeNet: A Benchmark for Molecular Machine Learning. *Chemical Science*, 9(2):513-530, 2018.

<a id="2">[2]</a> Aviv Navon, Aviv Shamsian, Idan Achituve, Haggai Maron, Kenji Kawaguchi, Gal Chechik, and Ethan Fetaya. Multi-task Learning as a Bargaining Game. In *International Conference on Machine Learning*, 2022.
