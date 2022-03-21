## Overall Framework

``LibMTL`` provides a unified framework to train a MTL model with several architectures and weighting strategies on benchmark datasets. The overall framework consists of nine modules as introduced below.

```eval_rst
- The Dataloader module is responsible for data pre-processing and loading.

- The `LibMTL.loss <../_autoapi/LibMTL/loss/index.html>`_ module defines loss functions for each task.

- The `LibMTL.metrics <../_autoapi/LibMTL/metrics/index.html>`_ module defines evaluation metrics for all the tasks.

- The `LibMTL.config <../_autoapi/LibMTL/config/index.html>`_ module is responsible for all the configuration parameters involved in the training process, such as the corresponding MTL setting (i.e. the multi-input case or not), the potential hyper-parameters of loss weighting strategies and architectures, the training configuration (e.g., the batch size, the running epoch, the random seed, and the learning rate), and so on. This module adopts command-line arguments to enable users to conveniently set those configuration parameters.

- The `LibMTL.Trainer <../_autoapi/LibMTL/trainer/index.html>`_ module provides a unified framework for the training process under different MTL settings and for different MTL approaches

- The `LibMTL.utils <../_autoapi/LibMTL/utils/index.html>`_ module implements some useful functionalities for the training process such as calculating the total number of parameters in an MTL model.

- The `LibMTL.architecture <../_autoapi/LibMTL/architecture/index.html>`_ module contains the implementations of various architectures in MTL.

- The `LibMTL.weighting <../_autoapi/LibMTL/weighting/index.html>`_ module contains the implementations of various loss weighting strategies in MTL.

- The `LibMTL.model <../_autoapi/LibMTL/model/index.html>`_ module includes some popular backbone networks (e.g., ResNet).
```

```eval_rst
.. figure:: ../images/framework.png
        :scale: 50%
```

