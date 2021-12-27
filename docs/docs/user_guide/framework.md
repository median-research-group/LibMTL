## Overall Framework

``LibMTL`` provides a unified running framework to train a MTL model with some kind of architectures or weighting strategies on a given dataset. The overall framework is shown below. There are five modules to support to run. We introduce them as follows.

- **Config Module**: Responsible for all the configuration parameters involved in the running framework, including the parameters of optimizer and learning rate scheduler, the hyper-parameters of MTL model, training configuration like batch size, total epoch, random seed and so on.
- **Dataloaders Module**: Responsible for data pre-processing and loading.
- **Model Module**: Responsible for inheriting classes architecture and weighting and instantiating a MTL model. Note that the architecture and the weighting strategy determine the forward and backward processes of the MTL model, respectively.
- **Losses Module**: Responsible for computing the loss for each task. 
- **Metrics Module**: Responsible for evaluating the MTL model and calculating the metric scores for each task.

```eval_rst
.. figure:: ../images/framework.png
        :scale: 100%
```

