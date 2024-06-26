import os
from typing import Union, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor

def stack_and_pad_same_context_size(x: List[Float[Tensor, "context feature"]]) \
        -> Float[Tensor, "batch context feature"]:
    """
    Stacks and pads a list of tensors to the same context size.
    Returns the size of the context original dimensions.
    Args:
        x: List of tensors. WIth different context sizes.

    Returns:

    """
    context_sizes = [elem.size(-2) for elem in x]
    max_context_size = max(context_sizes)
    batch_size = len(x)
    feature_size = x[0].size(-1)
    stacked = torch.zeros((batch_size, max_context_size, feature_size), dtype=x[0].dtype, device=x[0].device)
    for i, elem in enumerate(x):
        stacked[i, :context_sizes[i], :] = elem
    return stacked


def to_numpy(x: torch.Tensor) -> np.array:
    return x.squeeze().detach().cpu().numpy()


def t2np(x: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        raise Exception("Not a Array")


class BatchLinear(nn.Linear):
    """Helper class for linear layers on order-3 tensors.
    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Use a bias. Defaults to `True`.
    """

    def __init__(self, in_features, out_features, bias=True):
        super(BatchLinear, self).__init__(in_features=in_features,
                                          out_features=out_features,
                                          bias=bias)
        nn.init.xavier_normal_(self.weight, gain=1.0)
        if bias:
            nn.init.constant_(self.bias, 1e-3)

    def forward(self, x):
        """Forward pass through layer. First unroll batch dimension, then pass
        through dense layer, and finally reshape back to a order-3 tensor.
        Args:
              x (tensor): Inputs of shape `(batch, n, in_features)`.
        Returns:
              tensor: Outputs of shape `(batch, n, out_features)`.
        """
        num_functions, num_inputs = x.shape[0], x.shape[1]
        x = x.view(num_functions * num_inputs, self.in_features)
        out = super(BatchLinear, self).forward(x)
        return out.view(num_functions, num_inputs, self.out_features)


def to_multiple(x, multiple):
    """Convert `x` to the nearest above multiple.
    Args:
        x (number): Number to round up.
        multiple (int): Multiple to round up to.
    Returns:
        number: `x` rounded to the nearest above multiple of `multiple`.
    """
    if x % multiple == 0:
        return x
    else:
        return x + multiple - x % multiple


def init_sequential_weights(model, bias=0.0):
    """Initialize the weights of a nn.Sequential model with Glorot
    initialization.
    Args:
        model (:class:`nn.Module`): Container for model.
        bias (float, optional): Value for initializing bias terms. Defaults
            to `0.0`.
    Returns:
        (nn.Module): model with initialized weights
    """
    for layer in model:
        if hasattr(layer, 'weight'):
            nn.init.xavier_normal_(layer.weight, gain=1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, bias)
    return model


def compute_dists(x, y):
    """Fast computation of pair-wise distances for the 1d case.
    Args:
        x (tensor): Inputs of shape `(batch, n, 1)`.
        y (tensor): Inputs of shape `(batch, m, 1)`.
    Returns:
        tensor: Pair-wise distances of shape `(batch, n, m)`.
    """
    assert x.shape[2] == 1 and y.shape[2] == 1, \
        'The inputs x and y must be 1-dimensional observations.'
    return (x - y.permute(0, 2, 1)) ** 2


def init_layer_weights(layer):
    """Initialize the weights of a :class:`nn.Layer` using Glorot
    initialization.
    Args:
        layer (:class:`nn.Sequential`): Single dense or convolutional layer from
            :mod:`torch.nn`.
    Returns:
        :class:`nn.Sequential`: Single dense or convolutional layer with
            initialized weights.
    """
    nn.init.xavier_normal_(layer.weight, gain=1)
    nn.init.constant_(layer.bias, 1e-3)


def pad_concat(t1, t2):
    """Concat the activations of two layer channel-wise by padding the layer
    with fewer points with zeros.
    Args:
        t1 (tensor): Activations from first layers of shape `(batch, n1, c1)`.
        t2 (tensor): Activations from second layers of shape `(batch, n2, c2)`.
    Returns:
        tensor: Concatenated activations of both layers of shape
            `(batch, max(n1, n2), c1 + c2)`.
    """
    if t1.shape[2] > t2.shape[2]:
        padding = t1.shape[2] - t2.shape[2]
        if padding % 2 == 0:  # Even difference
            t2 = F.pad(t2, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t2 = F.pad(t2, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')
    elif t2.shape[2] > t1.shape[2]:
        padding = t2.shape[2] - t1.shape[2]
        if padding % 2 == 0:  # Even difference
            t1 = F.pad(t1, (int(padding / 2), int(padding / 2)), 'reflect')
        else:  # Odd difference
            t1 = F.pad(t1, (int((padding - 1) / 2), int((padding + 1) / 2)),
                       'reflect')

    return torch.cat([t1, t2], dim=1)


def sample_with_reparameterisation_trick(z_mu: torch.Tensor, z_sigma: torch.Tensor) -> torch.Tensor:
    mb_size, Z_dim = z_mu.shape[0], z_mu.shape[1]
    eps = torch.randn(mb_size, Z_dim)
    return z_mu + torch.exp(z_sigma / 2) * eps


def delete_artificial_batch_length(task: dict):
    return {k: v.squeeze(0)
            for k, v in task.items()}


def save_torch_model(model: nn.Module, name: str, path: str):
    name = name + '.pt' if not '.pt' in name else name
    save_name = os.path.join(path, name)
    torch.save(model, save_name)


class RunningAverage:
    """Maintain a running average."""

    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def reset(self):
        """Reset the running average."""
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        """Update the running average.
        Args:
            val (float): Value to update with.
            n (int): Number elements used to compute `val`.
        """
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def get_exponential_decay_gamma(scheduling_factor, max_epochs):
    """Return the exponential learning rate factor gamma.
    Parameters
    ----------
    scheduling_factor :
        By how much to reduce learning rate during training.
    max_epochs : int
        Maximum number of epochs.
    """
    return (1 / scheduling_factor) ** (1 / max_epochs)


def get_activation_name(activation):
    """Given a string or a `torch.nn.modules.activation` return the name of the activation."""
    if isinstance(activation, str):
        return activation

    mapper = {
        nn.LeakyReLU: "leaky_relu",
        nn.ReLU: "relu",
        nn.Tanh: "tanh",
        nn.Sigmoid: "sigmoid",
        nn.Softmax: "sigmoid",
    }
    for k, v in mapper.items():
        if isinstance(activation, k):
            return k

    raise ValueError("Unkown given activation type : {}".format(activation))


def get_gain(activation):
    """Given an object of `torch.nn.modules.activation` or an activation name
    return the correct gain."""
    if activation is None:
        return 1

    activation_name = get_activation_name(activation)

    param = None if activation_name != "leaky_relu" else activation.negative_slope
    gain = nn.init.calculate_gain(activation_name, param)

    return gain


def linear_init(module, activation="relu"):
    """Initialize a linear layer.
    Parameters
    ----------
    module : nn.Module
       module to initialize.
    activation : `torch.nn.modules.activation` or str, optional
        Activation that will be used on the `module`.
    """
    x = module.weight

    if module.bias is not None:
        module.bias.data.zero_()

    if activation is None:
        return nn.init.xavier_uniform_(x)

    activation_name = get_activation_name(activation)

    if activation_name == "leaky_relu":
        a = 0 if isinstance(activation, str) else activation.negative_slope
        return nn.init.kaiming_uniform_(x, a=a, nonlinearity="leaky_relu")
    elif activation_name == "relu":
        return nn.init.kaiming_uniform_(x, nonlinearity="relu")
    elif activation_name in ["sigmoid", "tanh"]:
        return nn.init.xavier_uniform_(x, gain=get_gain(activation))
