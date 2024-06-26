from typing import Tuple, Sequence

import torch
from jaxtyping import Float
from torch import nn, concatenate
from torch.nn import ModuleList
from typeguard import typechecked

from src.models.backbone_model.mlp import MLP


class DeepSet(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, hidden_layers: int = 2,
                 batch_norm: bool = False, activation = nn.ReLU(), hidden_size_factor: int = 2):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.batch_norm = batch_norm

        self.hidden_dim = max(32, hidden_size_factor * latent_dim)
        self.phi = MLP(input_size=self.input_dim + self.output_dim,
                       output_size=self.latent_dim,
                       hidden_size=self.hidden_dim,
                       n_hidden_layers=hidden_layers,
                       batch_norm=self.batch_norm,
                       activation=activation)

        self.pooling = MeanPooling(pooling_dim=-2)

    def forward(self,
                x_context: torch.Tensor,
                y_context: torch.Tensor,
                context_sizes: Sequence[int] = None) -> \
            torch.Tensor:
        decoder_input = concatenate([x_context, y_context], dim=-1)
        ri = self.phi(decoder_input)
        r = self.pooling(ri, context_sizes)
        return r


class MeanPooling(nn.Module):
    """Helper class for performing mean pooling in CNPs.
    Args:
        pooling_dim (int, optional): Dimension to pool over. Defaults to `0`.
    """

    def __init__(self, pooling_dim=0):
        super(MeanPooling, self).__init__()
        self.pooling_dim = pooling_dim
        assert pooling_dim == -2, "Only implemented for -2"

    def forward(self, h, context_sizes: Sequence[int] = None):
        """Perform pooling operation.
        context_sizes give the number of context points for each batch element.
        only these should be used for the mean
        Args:
            h: Tensor to pool over.
        """
        if context_sizes is not None:
            tmp = []
            for i, h_i in enumerate(h):
                part = h_i[:context_sizes[i]]
                tmp.append(torch.mean(part, dim=self.pooling_dim, keepdim=False))
            h = torch.stack(tmp, dim=0)
        else:
            h = torch.mean(h, dim=self.pooling_dim, keepdim=False)
        return h
