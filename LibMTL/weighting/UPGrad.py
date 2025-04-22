# The code of this file was partly taken from https://github.com/TorchJD/torchjd.
# It is therefore also subject to the following license:
#
# MIT License
#
# Copyright (c) Valérian Rey, Pierre Quinton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
from qpsolvers import solve_qp
from torch import Tensor

from LibMTL.weighting.abstract_weighting import AbsWeighting


class UPGrad(AbsWeighting):
    r"""Unconflicting Projection of Gradients (UPGrad).
    
    This method is proposed in `Jacobian Descent for Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_
    and implemented by adapting from the `official TorchJD implementation <https://github.com/TorchJD/torchjd>`_.
    """

    def __init__(self):
        super().__init__()
        
    def backward(self, losses, **kwargs):
        norm_eps = kwargs['UPGrad_norm_eps']
        reg_eps = kwargs['UPGrad_reg_eps']

        if self.rep_grad:
            raise ValueError('No support for method UPGrad with representation gradients (rep_grad=True)')
        else:
            self._compute_grad_dim()
            grads = self._compute_grad(losses, mode='backward')

        # Compute the weights corresponding to UPGrad
        pref_vector = torch.ones(len(grads), device=grads.device, dtype=grads.dtype) / len(grads)
        U = torch.diag(pref_vector)
        G = _regularize(_normalize(_compute_gramian(grads), norm_eps), reg_eps)
        W = _project_weights(U, G)
        weights = torch.sum(W, dim=0)

        # Store in the .grad fields the weighted sum of the gradients
        self._backward_new_grads(weights, grads=grads)

        # Return the weights as a numpy vector
        return weights.detach().cpu().numpy()


def _compute_gramian(matrix: Tensor) -> Tensor:
    """
    Computes the `Gramian matrix <https://en.wikipedia.org/wiki/Gram_matrix>`_ of a given matrix.
    """

    return matrix @ matrix.T


def _normalize(gramian: Tensor, eps: float) -> Tensor:
    """
    Normalizes the gramian `G=AA^T` with respect to the Frobenius norm of `A`.

    If `G=A A^T`, then the Frobenius norm of `A` is the square root of the trace of `G`, i.e., the
    sqrt of the sum of the diagonal elements. The gramian of the (Frobenius) normalization of `A` is
    therefore `G` divided by the sum of its diagonal elements.
    """
    squared_frobenius_norm = gramian.diagonal().sum()
    if squared_frobenius_norm < eps:
        return torch.zeros_like(gramian)
    else:
        return gramian / squared_frobenius_norm


def _regularize(gramian: Tensor, eps: float) -> Tensor:
    """
    Adds a regularization term to the gramian to enforce positive definiteness.

    Because of numerical errors, `gramian` might have slightly negative eigenvalue(s). Adding a
    regularization term which is a small proportion of the identity matrix ensures that the gramian
    is positive definite.
    """

    regularization_matrix = eps * torch.eye(gramian.shape[0], dtype=gramian.dtype, device=gramian.device)
    return gramian + regularization_matrix


def _project_weights(U: Tensor, G: Tensor) -> Tensor:
    """
    Computes the tensor of weights corresponding to the projection of the vectors in `U` onto the
    rows of a matrix whose Gramian is provided.

    :param U: The tensor of weights corresponding to the vectors to project, of shape `[..., m]`.
    :param G: The Gramian matrix of shape `[m, m]`. It must be symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    :return: A tensor of projection weights with the same shape as `U`.
    """

    G_ = _to_array(G)
    U_ = _to_array(U)

    W = np.apply_along_axis(lambda u: _project_weight_vector(u, G_), axis=-1, arr=U_)

    return torch.as_tensor(W, device=G.device, dtype=G.dtype)


def _project_weight_vector(u: np.ndarray, G: np.ndarray) -> np.ndarray:
    r"""
    Computes the weights `w` of the projection of `J^T u` onto the dual cone of the rows of `J`,
    given `G = J J^T` and `u`. In other words, this computes the `w` that satisfies
    `\pi_J(J^T u) = J^T w`, with `\pi_J` defined in Equation 3 of [1].

    By Proposition 1 of [1], this is equivalent to solving for `v` the following quadratic program:
    minimize        v^T G v
    subject to      u \preceq v

    Reference:
    [1] `Jacobian Descent For Multi-Objective Optimization <https://arxiv.org/pdf/2406.16232>`_.

    :param u: The vector of weights `u` of shape `[m]` corresponding to the vector `J^T u` to
        project.
    :param G: The Gramian matrix of `J`, equal to `J J^T`, and of shape `[m, m]`. It must be
        symmetric and positive definite.
    :param solver: The quadratic programming solver to use.
    """

    m = G.shape[0]
    w = solve_qp(G, np.zeros(m), -np.eye(m), -u, solver="quadprog")

    if w is None:  # This may happen when G has large values.
        raise ValueError("Failed to solve the quadratic programming problem.")

    return w


def _to_array(tensor: Tensor) -> np.ndarray:
    """Transforms a tensor into a numpy array with float64 dtype."""

    return tensor.cpu().detach().numpy().astype(np.float64)
