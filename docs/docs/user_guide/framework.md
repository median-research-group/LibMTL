## Overall Framework

``LibMTL`` provides a unified framework to train a MTL model with several architectures and weighting strategies on benchmark datasets. The overall framework consists of five modules as introduced below.

- **Config Module**: Responsible for all the configuration parameters involved in the framework, including the parameters of optimizer and learning rate scheduler, the hyper-parameters of MTL model, training configurations like batch size, total epoch, random seed and so on.
- **Dataloaders Module**: Responsible for data pre-processing and loading.
- **Model Module**: Responsible for inheriting classes architecture, weighting and instantiating a MTL model. Note that the architecture and the weighting strategy determine the forward and backward processes of the MTL model, respectively.
- **Losses Module**: Responsible for computing the loss for each task. 
- **Metrics Module**: Responsible for evaluating the MTL model and calculating the metric scores for each task.

```eval_rst
.. figure:: ../images/framework.png
        :scale: 100%
```

