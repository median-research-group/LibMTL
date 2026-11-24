import torch
import torch.nn as nn
import torch.nn.functional as F

from LibMTL.architecture.abstract_arch import AbsArchitecture


class _SigNorm(nn.Module):
    r"""A normalization layer with a sigmoid scale and no shift.

    The method drops the shift of a normalization layer and bounds its scale with a sigmoid: the
    output is ``sigmoid(gamma) * x_hat``, where ``x_hat`` is the normalized input. Only ``gamma``
    (``weight``) is learned, and ``sigmoid(gamma)`` is the importance of each channel. The method is
    meant to be trained from scratch, with ``gamma = 0``.

    Converting a pretrained layer is an addition to the method. A pretrained layer has a shift and
    scales outside ``(0, 1)``. To keep its function, the conversion reintroduces both as frozen
    constants: the output becomes ``k * sigmoid(gamma) * x_hat + beta``, where ``beta`` is the
    pretrained shift, ``k = max(1, max|scale| / 0.95)`` and ``gamma = logit(scale / k)``. Every
    pretrained scale above ``1e-3`` is reproduced. A negative scale cannot be, and becomes ``1e-3``.

    A layer still at its initial values (scale 1, shift 0) has not been trained. It gets the
    method's init: ``k = 1``, ``beta = 0`` and ``gamma = 0``.
    """
    def __init__(self, norm):
        super(_SigNorm, self).__init__()
        if norm.weight is None:
            raise ValueError('TSsNorm needs normalization layers with a learnable scale')
        scale = norm.weight.detach()
        shift = norm.bias.detach() if norm.bias is not None else torch.zeros_like(scale)
        if (scale == 1).all() and (shift == 0).all():  # not trained: the method's init
            scale = torch.full_like(scale, 0.5)
        k = max(1., scale.abs().max().item() / 0.95)
        self.weight = nn.Parameter((scale.clamp(min=1e-3) / k).logit())
        self.register_buffer('k', torch.tensor(k))
        self.register_buffer('bias', shift.clone())

    def importance(self):
        return torch.sigmoid(self.weight)

    def scale(self):
        return self.k * self.importance()


class _SigBN(_SigNorm):
    r""":class:`_SigNorm` for BatchNorm. It keeps the running statistics of the original layer."""
    def __init__(self, bn):
        super(_SigBN, self).__init__(bn)
        # holds the running statistics; its forward is not used, see forward below
        self.norm = type(bn)(bn.num_features, bn.eps, bn.momentum, affine=False,
                             track_running_stats=bn.track_running_stats)
        self.norm.load_state_dict(bn.state_dict(), strict=False)

    def forward(self, x):
        n = self.norm  # running statistics are updated as in torch's BatchNorm
        momentum = n.momentum
        if n.training and n.track_running_stats:
            n.num_batches_tracked.add_(1)
            momentum = momentum if momentum is not None else 1. / n.num_batches_tracked.item()
        stats = (n.running_mean, n.running_var) if not n.training or n.track_running_stats else (None, None)
        return F.batch_norm(x, *stats, self.scale(), self.bias,
                            n.training or n.running_mean is None, momentum or 0., n.eps)


class _SigLN(_SigNorm):
    r""":class:`_SigNorm` for LayerNorm."""
    def __init__(self, ln):
        super(_SigLN, self).__init__(ln)
        self.normalized_shape, self.eps = ln.normalized_shape, ln.eps

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.scale(), self.bias, self.eps)


class _TSNorm(nn.ModuleDict):
    r"""One :class:`_SigNorm` per task, applied for the active ``task``."""
    def __init__(self, sig_norm, norm, task_name):
        super(_TSNorm, self).__init__({task: sig_norm(norm) for task in task_name})
        self.task = task_name[0]

    def forward(self, x):
        return self[self.task](x)


_SIG_NORMS = {nn.BatchNorm1d: _SigBN, nn.BatchNorm2d: _SigBN, nn.BatchNorm3d: _SigBN, nn.LayerNorm: _SigLN}


def _replace_norms(module, task_name):
    for name, child in module.named_children():
        if type(child) in _SIG_NORMS:
            setattr(module, name, _TSNorm(_SIG_NORMS[type(child)], child, task_name))
        elif isinstance(child, (nn.modules.batchnorm._BatchNorm, nn.LayerNorm)):
            # a subclass may normalize differently, e.g. SyncBatchNorm
            raise ValueError('TSsNorm supports BatchNorm1d/2d/3d and LayerNorm, not {}'.format(type(child).__name__))
        else:
            _replace_norms(child, task_name)


class TSsNorm(AbsArchitecture):
    r"""Task-Specific Sigmoid Normalization (TSσNorm).

    This method is proposed in `Simplifying Multi-Task Architectures Through Task-Specific Normalization (TMLR 2026) <https://openreview.net/forum?id=QNO893OXrS>`_ \
    and implemented by modifying from the `official PyTorch implementation <https://github.com/xapharius/TSsNorm>`_.

    The encoder is shared, except for its BatchNorm and LayerNorm layers: each task gets its own,
    with a sigmoid scale and no shift (TSσBN and TSσLN). The scale is the importance of each channel
    for a task, see :func:`importance`. The method is meant to be trained from scratch. A pretrained
    encoder is converted so that it keeps its function, see :class:`_SigNorm`. The encoder runs once
    per task. As part of the method, the scales train at ``lr_mul`` times the learning rate and
    without weight decay, see :func:`get_param_groups`.
    """
    lr_mul = 100

    def __init__(self, task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs):
        super(TSsNorm, self).__init__(task_name, encoder_class, decoders, rep_grad, multi_input, device, **kwargs)
        self.encoder = self.encoder_class()
        _replace_norms(self.encoder, self.task_name)
        self.task_norms = [m for m in self.encoder.modules() if isinstance(m, _TSNorm)]
        if not self.task_norms:
            raise ValueError('TSsNorm found no BatchNorm or LayerNorm layer in the encoder')

    def forward(self, inputs, task_name=None):
        out = {}
        for tn, task in enumerate(self.task_name):
            if task_name is not None and task != task_name:
                continue
            for norm in self.task_norms:
                norm.task = task
            rep = self.encoder(inputs)
            rep = rep[tn] if isinstance(rep, list) else rep  # an encoder may return one rep per task
            out[task] = self.decoders[task](self._prepare_rep(rep, task, same_rep=False))
        return out

    def _task_params(self):
        return [p for norm in self.task_norms for p in norm.parameters()]

    def get_share_params(self):
        task_params = {id(p) for p in self._task_params()}
        return [p for p in self.encoder.parameters() if id(p) not in task_params]

    def zero_grad_share_params(self):
        for p in self.get_share_params():
            if p.grad is not None:
                p.grad.zero_()

    def get_param_groups(self, lr):
        r"""Return the optimizer's parameter groups: the scales at ``lr_mul * lr`` without weight
        decay, everything else with the optimizer's settings."""
        task_params = {id(p) for p in self._task_params()}
        return [{'params': [p for p in self.parameters() if id(p) not in task_params]},
                {'params': self._task_params(), 'lr': self.lr_mul * lr, 'weight_decay': 0.}]

    def importance(self):
        r"""Return the importance of each encoder channel for each task, a ``[tasks, channels]`` tensor."""
        return torch.stack([torch.cat([norm[task].importance().flatten() for norm in self.task_norms])
                            for task in self.task_name]).detach()
