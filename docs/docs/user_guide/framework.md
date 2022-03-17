## Overall Framework

``LibMTL`` provides a unified framework to train a MTL model with several architectures and weighting strategies on benchmark datasets. The overall framework consists of nine modules as introduced below.

```eval_rst
- Dataloaders Module: Responsible for data pre-processing and loading.

- `LibMTL.loss <../_autoapi/LibMTL/loss/index.html>`_: Responsible for computing the loss for each task. 

- `LibMTL.metrics <../_autoapi/LibMTL/metrics/index.html>`_: Responsible for evaluating an MTL model and calculating the metric scores for each task.

- `LibMTL.config <../_autoapi/LibMTL/config/index.html>`_: Responsible for all the configuration parameters involved in the framework, including the parameters of optimizer and learning rate scheduler, the hyper-parameters of MTL model, training configurations like batch size, total epoch, random seed and so on.

- `LibMTL.Trainer <../_autoapi/LibMTL/trainer/index.html>`_: Responsible for a unified training framework for MTL, covering both the single-input and multi-input problems.

- `LibMTL.utils <../_autoapi/LibMTL/utils/index.html>`_: Responsible for some useful functions in training, such as calculating the number of parameters of an MTL model.

- `LibMTL.architecture <../_autoapi/LibMTL/architecture/index.html>`_: Responsible for the implementations of various architectures in MTL.

- `LibMTL.weighting <../_autoapi/LibMTL/weighting/index.html>`_: Responsible for the implementations of various loss weighting strategies in MTL.

- `LibMTL.model <../_autoapi/LibMTL/model/index.html>`_: Responsible for the implementations of popular backbone networks, such as ResNet-50.
```

```eval_rst
.. figure:: ../images/framework.png
        :scale: 80%
```

