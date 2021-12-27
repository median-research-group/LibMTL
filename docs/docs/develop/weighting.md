## Customize a Weighting Strategy

Here we would like to introduce how to customize a new weighting strategy with the support of ``LibMTL``.

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

### Rewrite Corresponding Methods

```eval_rst
There are four important function in :class:`LibMTL.weighting.AbsWeighting`. We will introduce them in detail as follows.

- :func:`backward`: The main function of a weighting strategy its input and output format can be found in :func:`LibMTL.weighting.AbsWeighting.backward`. To rewrite this function, you need to consider the case of ``multi-input`` and ``multi-label`` (refer to `here <../user_guide/mtl.html#network-architecture>`_) and the case of ``rep-grad`` and ``param-grad`` (refer to `here <../user_guide/mtl.html#weighting-strategy>`_) if you would like to combine your weighting method with more architectures or apply your method to more datasets.
- :func:`init_param`: This function is used to define and initialize some trainable parameters. It does nothing by default and can be rewritten if necessary.
- :func:`_get_grads`: This function is used to return the gradients of representations or shared parameters (covering the case of ``rep-grad`` and ``param-grad``).
- :func:`_backward_new_grads`: This function is used to reset the gradients and make a backward (covering the case of ``rep-grad`` and ``param-grad``).

The :func:`_get_grads` and :func:`_backward_new_grads` functions are very useful to rewrite the :func:`backward` function and you can find more details about them in `here <../../_modules/LibMTL/weighting/abstract_weighting.html#AbsWeighting>`_.
```

