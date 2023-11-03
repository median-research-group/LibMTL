## PAWS-X from XTREME Benchmark

The PAWS-X dataset is a multilingual sentence classification dataset from XTREME benchmark [[1]](#1). Following [[2]](#2), we use English (en), Mandarin (zh), German (de) and Spanish (es) to form a multi-input multi-task problem. Each language/task has about 49.4K, 2.0K, and 2.0K data samples for training, validation, and testing. The training settings are mainly followed [[2]](#2) and the codes are modified from [xtreme](https://github.com/google-research/xtreme). 

Run the following command to download the dataset,

```shell
bash propocess_data/download_data.sh
```

### Dependencies

- networkx==1.11

- transformers==4.6.1

### Run a Model

The script ``main.py`` is the main file for training and evaluating a MTL model on the PAWS-X dataset. A set of command-line arguments is provided to allow users to adjust the training parameter configuration. 

Some important  arguments are described as follows.

- ``weighting``: The weighting strategy. Refer to [here](../../LibMTL#supported-algorithms).
- ``arch``: The MTL architecture. Refer to [here](../../LibMTL#supported-algorithms).
- ``gpu_id``: The id of gpu. The default value is '0'.
- ``seed``: The random seed for reproducibility. The default value is 0.
- ``dataset_path``: The path of the PAWS-X dataset.
- ``bs``: The batch size of training, validation, and test data. The default value is 32.

The complete command-line arguments and their descriptions can be found by running the following command.

```shell
python main.py -h
```

If you understand those command-line arguments, you can train a MTL model by running a command like this. 

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --multi_input --mode train --save_path PATH
```

You can test the trained MTL model by running the following command.

```shell
python main.py --weighting WEIGHTING --arch ARCH --dataset_path PATH --gpu_id GPU_ID --multi_input --mode test --load_path PATH
```

### References

<a id="1">[1]</a> Junjie Hu, Sebastian Ruder, Aditya Siddhant, Graham Neubig, Orhan Firat, and Melvin Johnson. XTREME: A Massively Multilingual Multi-task Benchmark for Evaluating Cross-lingual Generalisation. In *International Conference on Machine Learning*, 2020.

<a id="2">[2]</a> Baijiong Lin, Feiyang Ye, Yu Zhang, and Ivor Tsang. Reasonable Effectiveness of Random Weighting: A Litmus Test for Multi-task Learning. *Transactions on Machine Learning Research*, 2022.
