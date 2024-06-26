from typing import Tuple, List, Union, Sequence, Any

import torch
from jaxtyping import Float
from omegaconf import ListConfig
from torch import nn, Tensor
from torch.distributions import Distribution, Normal
from torch.nn import Parameter
from typeguard import typechecked

from src.models.backbone_model.mlp import MLP
from src.models.deep_set import DeepSet
from src.models.mlp_decoder import MLPDecoder
from src.modules.cnp import CNP


class NdCNP(CNP):
    """
    Basic CNP for a variable input and output size
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 latent_dim: int,
                 global_residual: bool = False,
                 batch_norm: bool = False,
                 size: str = "S",
                 batch_norm_xencoder: bool = False,
                 batch_norm_deepset: bool = False,
                 state_idx: Union[List[int], ListConfig[int]] = [],
                 x_encoder: bool = True,
                 activation: nn.Module = nn.ReLU(),
                 deepset_hidden_size_factor: int = 2,
                 *args, **kwargs):
        """
        Args:
            input_dim: Dimension of the input datapoints
            output_dim: Dimension of the output datapoints
            latent_dim: Latent dim that is used in all the MLPs
            global_residual: Whether to use a global residual connection
            batch_norm: Whether to use batch normalization in the decoder
            size: Size of the MLPs, either "S", "M" or "L"
            batch_norm_xencoder: Whether to use batch normalization in the x_encoder
            batch_norm_deepset: Whether to use batch normalization in the enconder (deepset)
        """
        super().__init__(*args, **kwargs)
        if not state_idx:
            self.state_idx = Parameter(torch.tensor(range(input_dim), dtype=torch.long), requires_grad=False)
        else:
            self.state_idx = Parameter(torch.tensor(state_idx, dtype=torch.long), requires_grad=False)
        self.input_dim = input_dim
        self.encoded_input_dim = input_dim + self.data_encoder_and_normalizer.extra_dims
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.global_residual = global_residual
        self.batch_norm = batch_norm
        self.size = size
        batch_norm_xencoder = batch_norm_xencoder
        batch_norm_deepset = batch_norm_deepset
        if x_encoder:
            x_encoded_input_dim = latent_dim
        else:
            x_encoded_input_dim = self.encoded_input_dim

        if self.diff_as_context_y:
            deppset_extra_output_dim = output_dim
        else:
            deppset_extra_output_dim = output_dim + self.data_encoder_and_normalizer.extra_dims

        if self.size == "S":
            self.x_encoder = MLP(input_size=self.encoded_input_dim, output_size=latent_dim, hidden_size=latent_dim, batch_norm=batch_norm_xencoder, activation=activation) if x_encoder else None
            self.encoder = DeepSet(x_encoded_input_dim, deppset_extra_output_dim, latent_dim,
                                   batch_norm=batch_norm_deepset, activation=activation, hidden_size_factor=deepset_hidden_size_factor)
            self.decoder = MLPDecoder(x_encoded_input_dim, 2 * self.output_dim, latent_dim, batch_norm, activation=activation)
        elif self.size == "M":
            self.x_encoder = MLP(input_size=self.encoded_input_dim, output_size=latent_dim, hidden_size=latent_dim,
                                 batch_norm=batch_norm_xencoder, n_hidden_layers=2, activation=activation) if x_encoder else None
            self.encoder = DeepSet(x_encoded_input_dim, deppset_extra_output_dim, latent_dim,
                                   batch_norm=batch_norm_deepset, hidden_layers=3, activation=activation, hidden_size_factor=deepset_hidden_size_factor)
            self.decoder = MLPDecoder(x_encoded_input_dim, 2 * self.output_dim, latent_dim, batch_norm, n_hidden_layer=6, activation=activation)
        elif self.size == "L":
            self.x_encoder = MLP(input_size=self.encoded_input_dim, output_size=latent_dim, hidden_size=latent_dim,
                                 batch_norm=batch_norm_xencoder, n_hidden_layers=2, activation=activation) if x_encoder else None
            self.encoder = DeepSet(x_encoded_input_dim, deppset_extra_output_dim, latent_dim,
                                   batch_norm=batch_norm_deepset, hidden_layers=4, activation=activation, hidden_size_factor=deepset_hidden_size_factor)
            self.decoder = MLPDecoder(x_encoded_input_dim, 2 * self.output_dim, latent_dim, batch_norm, n_hidden_layer=8, activation=activation)
        elif self.size == "S2":
            self.x_encoder = MLP(input_size=self.encoded_input_dim, output_size=latent_dim, hidden_size=latent_dim,
                                 batch_norm=batch_norm_xencoder, n_hidden_layers=1, activation=activation) if x_encoder else None
            self.encoder = DeepSet(x_encoded_input_dim, deppset_extra_output_dim, latent_dim,
                                   batch_norm=batch_norm_deepset, hidden_layers=3, activation=activation, hidden_size_factor=deepset_hidden_size_factor)
            self.decoder = MLPDecoder(x_encoded_input_dim, 2 * self.output_dim, latent_dim, batch_norm, n_hidden_layer=4, activation=activation)
        self.sigma_fn = nn.Softplus()  # Softplus to enforce positivity of the plot_std

        self.example_input_array = (torch.rand(3, 22, self.input_dim),
                                    torch.rand(3, 22, self.output_dim),
                                    torch.rand(3, 4, self.input_dim))

    @property
    def z_latent_dim(self) -> int:
        return self.latent_dim

    def get_context_encoding(self,
                             x_context_norm: torch.Tensor,
                             y_context_norm: torch.Tensor,
                             x_target_norm: torch.Tensor = None,
                             y_target_norm: torch.Tensor = None,
                             context_sizes: Sequence[int] = None,
                             use_latent_mean: bool = False,
                     **kwargs) -> \
            tuple[torch.Tensor, None, None]:

        # get vector embedding (DeepSet)
        en_x_context = self.x_encoder(x_context_norm) if self.x_encoder else x_context_norm
        r = self.encoder(en_x_context, y_context_norm, context_sizes=context_sizes)
        return r, None, None

    def get_diff_context_encoding(self,
                             x_context_norm: torch.Tensor,
                             diff_context_norm: torch.Tensor,
                             x_target_norm: torch.Tensor = None,
                             y_target_norm: torch.Tensor = None,
                             context_sizes: Sequence[int] = None,
                             use_latent_mean: bool = False,
                             **kwargs) -> \
            Tuple[torch.Tensor, Distribution, Distribution]:
        """
        Args:
            use_latent_mean:
            x_context_norm:
            diff_context_norm:
            x_target_norm:
            y_target_norm:
            context_sizes:

        Returns:
            z_sample: The used latent sample
            z_dist_c: The distribution of the latent space for the context set
            z_dist_ct: The distribution of the latent space for the target and context set

        """
        en_x_context = self.x_encoder(x_context_norm) if self.x_encoder else x_context_norm
        r = self.encoder(en_x_context, diff_context_norm, context_sizes=context_sizes)
        return r, None, None

    def get_decoding(self,
                hidden_encoding: torch.Tensor,
                x_target: torch.Tensor,
                x_target_norm: torch.Tensor,
                     **kwargs) -> \
            Distribution:

        en_x_target = self.x_encoder(x_target_norm) if self.x_encoder else x_target_norm

        assert hidden_encoding.size(0) == x_target.size(0), "Number of batches must be the same for hidden_encoding and x_target"

        decoder_input = torch.cat((hidden_encoding, en_x_target), dim=-1)
        y_pred = self.decoder(decoder_input)
        y_mean_pred, y_std_pred = y_pred.split(self.output_dim, dim=-1)
        y_std_pred = self.sigma_fn(y_std_pred) + torch.finfo(torch.float32).eps
        if self.global_residual:
            y_mean_pred = self.data_encoder_and_normalizer.normalize_increment(y_mean_pred)
            y_mean_pred = y_mean_pred + x_target[:, :, self.state_idx]
        else:
            y_mean_pred = self.data_encoder_and_normalizer.normalize_output(y_mean_pred)
        return Normal(loc=y_mean_pred, scale=y_std_pred)

    def repeat_for_target_size(self, r, target_size):
        return r.unsqueeze(1).repeat(1, target_size, 1)  # batch, target, feature