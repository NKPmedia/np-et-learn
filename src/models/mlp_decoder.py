from torch import nn

from src.models.backbone_model.mlp import MLP


class MLPDecoder(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, latent_dim: int, batch_norm: bool = False, n_hidden_layer: int = 4, activation = nn.ReLU()):
        super(MLPDecoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.n_hidden_layer = n_hidden_layer
        self.batch_norm = batch_norm

        self.decoder_test = MLP(input_size=self.latent_dim + self.input_dim,
                                output_size=self.output_dim,
                                hidden_size=self.latent_dim,
                                n_hidden_layers=n_hidden_layer,
                                batch_norm=batch_norm,
                                activation=activation)

    def forward(self, latent_representation):
        return self.decoder_test(latent_representation)
