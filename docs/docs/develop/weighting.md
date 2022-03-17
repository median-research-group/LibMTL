## Customize a Weighting Strategy

Here we introduce how to customize a new weighting strategy with the support of ``LibMTL``.

### Create a New Weighting Class

```eval_rst
Firstly, you need to create a new weighting class by inheriting class :class:`LibMTL.weighting.AbsWeighting`.
```

```python
from LibMTL.weighting import AbsWeighting

class NewWeighting(AbsWeighting):
    def __init__(self):
        super(NewWeighting, self).__init__()
```

### Rewrite Relevant Methods

```eval_rst
There are four important functions in :class:`LibMTL.weighting.AbsWeighting`.

- :func:`backward`: It is the main function of a weighting strategy whose input and output formats can be found in :func:`LibMTL.weighting.AbsWeighting.backward`. To rewrite this function, you need to consider the case of ``single-input`` and ``multi-input`` (refer to `here <../user_guide/mtl.html#network-architecture>`_) and the case of ``rep-grad`` and ``param-grad`` (refer to `here <../user_guide/mtl.html#weighting-strategy>`_) if you want to combine your weighting method with more architectures or apply your method to more datasets.
- :func:`init_param`: This function is used to define and initialize some trainable parameters. It does nothing by default and can be rewritten if necessary.
- :func:`_get_grads`: This function is used to return the gradients of representations or shared parameters (corresponding to the case of ``rep-grad`` and ``param-grad``, respectively).
- :func:`_backward_new_grads`: This function is used to reset the gradients and make a backward pass (corresponding to the case of ``rep-grad`` and ``param-grad``, respectively).

The :func:`_get_grads` and :func:`_backward_new_grads` functions are very useful to rewrite the :func:`backward` function and you can find more details `here <../../_modules/LibMTL/weighting/abstract_weighting.html#AbsWeighting>`_.
```

