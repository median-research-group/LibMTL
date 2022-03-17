## Customize an Architecture

Here we introduce how to customize a new architecture with the support of ``LibMTL``.

### Create a New Architecture Class

```eval_rst
Firstly, you need to create a new architecture class by inheriting class :class:`LibMTL.architecture.AbsArchitecture`.
```

```python
from LibMTL.architecture import AbsArchitecture

class NewArchitecture(AbsArchitecture):
    def __init__(self, task_name, encoder_class, decoders, rep_grad, 
                       multi_input, device, **kwargs):
        super(NewArchitecture, self).__init__(task_name, encoder_class, decoders, rep_grad, 
                                  			  multi_input, device, **kwargs)
```

### Rewrite Relevant Methods

```eval_rst
There are four important functions in :class:`LibMTL.architecture.AbsArchitecture`.

- :func:`forward`: The forward function and its input/output format can be found in :func:`LibMTL.architecture.AbsArchitecture.forward`. To rewrite this function, you need to consider the case of ``single-input`` and ``multi-input`` (refer to `here <../user_guide/mtl.html#network-architecture>`_) and the case of ``rep-grad`` and ``param-grad`` (refer to `here <../user_guide/mtl.html#weighting-strategy>`_) if you want to combine your architecture with more weighting strategies or apply your architecture to more datasets.
- :func:`get_share_params`: This function is used to return the shared parameters of the model. It returns all the parameters of the encoder by default. You can rewrite it if necessary.
- :func:`zero_grad_share_params`: This function is used to set gradients of the shared parameters to zero. It will set the gradients of all the encoder parameters to zero by default. You can rewrite it if necessary.
- :func:`_prepare_rep`: This function is used to compute the gradients for representations. More details can be found `here <../../_modules/LibMTL/architecture/abstract_arch.html#AbsArchitecture>`_.
```



