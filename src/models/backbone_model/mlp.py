import warnings

import torch
import torch.nn as nn
from jaxtyping import Float
from torch.nn import BatchNorm1d
from typeguard import typechecked

from src.utils.torch_utils import linear_init


class MLP(nn.Module):
    """General MLP class.
    Parameters
    ----------
    input_size: int
    output_size: int
    hidden_size: int, optional
        Number of hidden neurones.
    n_hidden_layers: int, optional
        Number of hidden layers.
    activation: callable, optional
        Activation function. E.g. `nn.RelU()`.
    is_bias: bool, optional
        Whether to use biaises in the hidden layers.
    dropout: float, optional
        Dropout rate.
    is_force_hid_smaller : bool, optional
        Whether to force the hidden dimensions to be smaller or equal than in and out.
        If not, it forces the hidden dimension to be larger or equal than in or out.
    is_res : bool, optional
        Whether to use residual connections.
    """

    def __init__(
            self,
            input_size,
            output_size,
            hidden_size=32,
            n_hidden_layers=1,
            activation=nn.ReLU(),
            has_bias=True,
            dropout=0,
            is_res=False,
            batch_norm=False
    ):
        super().__init__()
        assert (isinstance(activation, torch.nn.ReLU) or
                isinstance(activation, torch.nn.Tanh) or
                isinstance(activation, torch.nn.SiLU)\
                ), "Only ReLU, Tanh and SiLU are supported as activation functions"
        self.input_size = input_size
        self.output_size = output_size
        if isinstance(hidden_size, int):
            self.hidden_size = [hidden_size] * n_hidden_layers
        else:
            self.hidden_size = hidden_size
        self.n_hidden_layers = n_hidden_layers
        self.is_res = is_res



        if batch_norm:
            notOut_has_bias = False
            print("WARNING: Batch norm is used, bias of FC-Layer is set to False")
        else:
            notOut_has_bias = has_bias

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.activation = activation
        self.batch_norm = batch_norm

        self.to_hidden = nn.Linear(self.input_size, self.hidden_size[0], bias=notOut_has_bias)

        self.linears = nn.ModuleList(
            [
                nn.Linear(layer_hidden_size_from, layer_hidden_size_to, bias=notOut_has_bias)
                for layer_hidden_size_from, layer_hidden_size_to in zip(self.hidden_size[:-1], self.hidden_size[1:])
            ]
        )
        self.to_hidden_batch_norm = BatchNorm1d(self.hidden_size[0])
        self.batch_norms = nn.ModuleList(
            [
                BatchNorm1d(layer_hidden_size)
                for layer_hidden_size in self.hidden_size[1:]
            ]
        )
        self.out = nn.Linear(self.hidden_size[-1], self.output_size, bias=has_bias)

    def forward(self, x: torch.Tensor):
        tmp = self.to_hidden(x)
        n_batch = tmp.shape[0]
        if self.batch_norm:
            tmp = self.to_hidden_batch_norm(tmp.view((-1, self.hidden_size[0]))).view((n_batch, -1, self.hidden_size[0]))
        tmp = self.activation(tmp)
        tmp = self.dropout(tmp)

        for linear, batchnorm, layer_hidden_size in zip(self.linears, self.batch_norms, self.hidden_size[1:]):
            diff = linear(tmp)
            if self.batch_norm:
                n_batch = diff.shape[0]
                diff = batchnorm(diff.view((-1, layer_hidden_size))).view((n_batch, -1, layer_hidden_size))
            diff = self.activation(diff)
            if self.is_res:
                tmp = tmp + diff
            else:
                tmp = diff
            tmp = self.dropout(tmp)

        out = self.out(tmp)
        return out
