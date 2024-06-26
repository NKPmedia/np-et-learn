from typing import Dict, Tuple, List, Sequence, Union

import torch
from jaxtyping import Float
from omegaconf import ListConfig, OmegaConf
from torch import Tensor


class NoiseAdder:

    def sample_noise_lvl(self):
        pass

    def add_noise_to_data(self,
                          x_context: Float[Tensor, "batch context in"],
                          y_context: Float[Tensor, "batch context out"],
                          x_target: Float[Tensor, "batch target in"],
                          y_target: Float[Tensor, "batch target out"]) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"],
            Float[Tensor, "batch target in"], Float[Tensor, "batch target out"]]:
        """
        Abstract method to add noise to data.
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)
            x_target: shape (batch_size, n_target, input_dim)
            y_target: shape (batch_size, n_target, output_dim)

        Returns:

        """

    def add_noise_context(self,
                          x_context: Float[Tensor, "batch context in"],
                          y_context: Float[Tensor, "batch context out"]) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"]]:
        """
        Abstract method to add noise to context data
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)

        Returns:

        """
        raise NotImplementedError

    def add_noise_target(self,
                         x_target: Float[Tensor, "batch target in"],
                         y_target: Float[Tensor, "batch target out"]) -> \
            Tuple[Float[Tensor, "batch target in"], Float[Tensor, "batch target out"]]:
        """
        Abstract method to add noise to target data
        Args:
            x_target: shape (batch_size, n_target, input_dim)
            y_target: shape (batch_size, n_target, output_dim)

        Returns:

        """
        raise NotImplementedError


class ProcessAndObservationNoiseAdder(NoiseAdder):
    """Adds procces and observation gaussian noise to context and target data"""

    def __init__(self,
                 observation_context_noise_level: Union[float, Sequence[float]],
                 observation_target_noise_level: Union[float, Sequence[float]],
                 process_context_noise_level: Union[float, Sequence[float]],
                 process_target_noise_level: Union[float, Sequence[float]],
                 equal_noise_lvls: bool = True,
                 state_idx: Sequence[int] = None,
                 sequence_for_states: bool = False
                 ):
        """
        Args:
            observation_context_noise_level:
            observation_target_noise_level:
            process_context_noise_level:
            process_target_noise_level:
        """
        if equal_noise_lvls:
            assert observation_target_noise_level == observation_context_noise_level, \
                "If equal_noise_lvls is True, observation_target_noise_level and observation_context_noise_level must be equal"
            assert process_target_noise_level == process_context_noise_level, \
                "If equal_noise_lvls is True, process_target_noise_level and process_context_noise_level must be equal"

        self.observation_context_noise_level = OmegaConf.to_container(observation_context_noise_level) if isinstance(
            observation_context_noise_level, ListConfig) else observation_context_noise_level
        self.observation_target_noise_level = OmegaConf.to_container(observation_target_noise_level) if isinstance(
            observation_target_noise_level, ListConfig) else observation_target_noise_level
        self.process_context_noise_level = OmegaConf.to_container(process_context_noise_level) if isinstance(
            process_context_noise_level, ListConfig) else process_context_noise_level
        self.process_target_noise_level = OmegaConf.to_container(process_target_noise_level) if isinstance(
            process_target_noise_level, ListConfig) else process_target_noise_level

        self.equal_noise_lvls = equal_noise_lvls
        self.state_idx = state_idx
        if isinstance(state_idx, ListConfig):
            self.state_idx = OmegaConf.to_container(state_idx)

        self.sequence_for_states = sequence_for_states

        self.sample_noise_lvl()

    def sample_noise_lvl(self):
        """
        Goes through all the noise levels and samples a noise level uniformly from the interval
        If the noise level is not a Sequence, it takes the value as the noise level
        if equal_noise_lvls is True it takes the same noise level for target and context
        Returns:

        """
        if isinstance(self.observation_context_noise_level, Sequence) and not self.sequence_for_states:
            self.sampled_observation_context_noise_level = (torch.rand(1) * (
                    self.observation_context_noise_level[1] - self.observation_context_noise_level[0]) +
                                                           self.observation_context_noise_level[0]).item()
        else:
            self.sampled_observation_context_noise_level = self.observation_context_noise_level

        if isinstance(self.observation_target_noise_level, Sequence) and not self.sequence_for_states:
            self.sampled_observation_target_noise_level = (torch.rand(1) * (
                    self.observation_target_noise_level[1] - self.observation_target_noise_level[0]) +
                                                          self.observation_target_noise_level[0]).item()
        else:
            self.sampled_observation_target_noise_level = self.observation_target_noise_level

        if isinstance(self.process_context_noise_level, Sequence) and not self.sequence_for_states:
            self.sampled_process_context_noise_level = (torch.rand(1) * (
                    self.process_context_noise_level[1] - self.process_context_noise_level[0]) +
                                                       self.process_context_noise_level[0]).item()
        else:
            self.sampled_process_context_noise_level = self.process_context_noise_level

        if isinstance(self.process_target_noise_level, Sequence) and not self.sequence_for_states:
            self.sampled_process_target_noise_level = (torch.rand(1) * (
                    self.process_target_noise_level[1] - self.process_target_noise_level[0]) +
                                                      self.process_target_noise_level[0]).item()
        else:
            self.sampled_process_target_noise_level = self.process_target_noise_level

        if self.equal_noise_lvls:
            self.sampled_observation_target_noise_level = self.sampled_observation_context_noise_level
            self.sampled_process_target_noise_level = self.sampled_process_context_noise_level

    def add_noise_to_data(self,
                          x_context: Float[Tensor, "batch context in"],
                          y_context: Float[Tensor, "batch context out"],
                          x_target: Float[Tensor, "batch target in"],
                          y_target: Float[Tensor, "batch target out"]) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"],
            Float[Tensor, "batch target in"], Float[Tensor, "batch target out"]]:
        """
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)
            x_target: shape (batch_size, n_target, input_dim)
            y_target: shape (batch_size, n_target, output_dim)

        Returns:

        """
        x_context, y_context = self.add_observation_process_noise(
            x_context,
            y_context,
            self.sampled_observation_context_noise_level,
            self.sampled_process_context_noise_level)
        x_target, y_target = self.add_observation_process_noise(
            x_target,
            y_target,
            self.sampled_observation_target_noise_level,
            self.sampled_process_target_noise_level)
        return x_context, y_context, x_target, y_target

    def add_noise_context(self,
                          x_context: Float[Tensor, "batch context in"],
                          y_context: Float[Tensor, "batch context out"]) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"]]:
        return self.add_observation_process_noise(
            x_context,
            y_context,
            self.sampled_observation_context_noise_level,
            self.sampled_process_context_noise_level)

    def add_noise_target(self,
                         x_target: Float[Tensor, "batch target in"],
                         y_target: Float[Tensor, "batch target out"]) -> \
            Tuple[Float[Tensor, "batch target in"], Float[Tensor, "batch target out"]]:
        return self.add_observation_process_noise(
            x_target,
            y_target,
            self.sampled_observation_target_noise_level,
            self.sampled_process_target_noise_level)

    def add_observation_noise(self, x, noise_lvl: float = None):
        """
        Adds gaussian observation noise to x
        Args:
            x: shape (batch_size, n, input_dim)
            noise_lvl: if None, uses self.observation_context_noise_level (observation and process noise have to be equal)
        Returns:

        """
        if noise_lvl is None:
            assert self.sampled_observation_context_noise_level == self.sampled_observation_target_noise_level, \
                "Process noise levels must be equal if noise_lvl is not specified"
            noise_lvl = self.sampled_observation_context_noise_level
        if self.state_idx:
            if self.sequence_for_states:
                std = torch.tensor(noise_lvl, device=x.device)[None]
                std = std.tile((x.shape[0], 1))
                mean = torch.zeros_like(std)
                x[..., self.state_idx] = x[..., self.state_idx] + torch.normal(mean, std)
            else:
                x[..., self.state_idx] = x[..., self.state_idx] + torch.normal(0.,noise_lvl,
                                                                               size=x[..., self.state_idx].shape,
                                                                               device=x.device)
        else:
            x = x + torch.normal(0., noise_lvl, size=x.shape, device=x.device)
        return x

    def add_process_noise(self, x, y, noise_lvl: float = None):
        """
        Adds gaussian process noise to y
        Process noise is scaled gaussian noise based on the magnitude of difference between x and y
        Args:
            x: shape (batch_size, n, input_dim)
            y: shape (batch_size, n, output_dim)
            noise_lvl: if None, uses self.observation_context_noise_level (observation and process noise have to be equal)

        Returns:

        """
        if noise_lvl is None:
            assert self.sampled_process_target_noise_level == self.sampled_process_context_noise_level, \
                "Process noise levels must be equal if noise_lvl is not specified"
            noise_lvl = self.sampled_process_target_noise_level
        if self.state_idx:
            y = y + torch.normal(0., noise_lvl, size=y.shape, device=y.device) * torch.abs(x[..., self.state_idx] - y)
        else:
            y = y + torch.normal(0., noise_lvl, size=y.shape, device=y.device) * torch.abs(x - y)
        return y

    def add_observation_process_noise(self, x, y, observation_lvl: float = None, process_lvl: float = None) -> \
            Tuple[Float[Tensor, "batch _ in"], Float[Tensor, "batch _ out"]]:
        """
        Adds observation and process noise to data
        Observation noise is homogeneous gaussian noise on x and y
        Process noise is scaled gaussian noise based on the magnitude of difference between x and y
        Process noise is only applied to y
        Args:
            x: shape (batch_size, n, input_dim)
            y: shape (batch_size, n, output_dim)
            observation_lvl:
            process_lvl:

        Returns:

        """
        y = self.add_process_noise(x, y, process_lvl)
        x = self.add_observation_noise(x, observation_lvl)
        y = self.add_observation_noise(y, observation_lvl)
        return x, y


class ZeroNoiseAdder(ProcessAndObservationNoiseAdder):
    """Like ProcessAndObservationNoiseAdder but allways adds no noise"""

    def __init__(self):
        super().__init__(0., 0., 0., 0.)

    def add_process_noise(self, x, y, noise_lvl: float = None):
        return y


class HomogeneousGaussianNoiseAdder(NoiseAdder):
    """Adds homogeneous gaussian noise to context and target data"""

    def __init__(self,
                 context_noise_level: float,
                 target_noise_level: float,
                 context_apply_to: str = "y",
                 target_apply_to: str = "y"
                 ):
        """
        Args:
            context_noise_level: Noise variance of the gaussian noise
            target_noise_level: Noise variance of the gaussian noise
            context_apply_to: Where to apply the noise to can be "x", "y", "both"
            target_apply_to: Where to apply the noise to can be "x", "y", "both"
        """
        self.context_noise_level = context_noise_level
        self.target_noise_level = target_noise_level
        self.context_apply_to = context_apply_to
        self.target_apply_to = target_apply_to

    def sample_noise_lvl(self):
        raise NotImplementedError

    def add_noise_to_data(self,
                          x_context: Float[Tensor, "batch context in"],
                          y_context: Float[Tensor, "batch context out"],
                          x_target: Float[Tensor, "batch target in"],
                          y_target: Float[Tensor, "batch target out"]) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"],
            Float[Tensor, "batch target in"], Float[Tensor, "batch target out"]]:
        """
        Adds homogeneous gaussian noise to context and target data
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)
            x_target: shape (batch_size, n_target, input_dim)
            y_target: shape (batch_size, n_target, output_dim)

        Returns:

        """
        x_context, y_context = self.add_noise_context(x_context, y_context)
        x_target, y_target = self.add_noise_target(x_target, y_target)
        return x_context, y_context, x_target, y_target

    def add_noise_context(self, x_context, y_context) -> \
            Tuple[Float[Tensor, "batch context in"], Float[Tensor, "batch context out"]]:
        """
        Adds homogeneous gaussian noise to context data
        Noise can be applied to x, y or both
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)

        Returns:

        """
        if self.context_apply_to == "x" or self.context_apply_to == "both":
            x_context += torch.normal(0, self.context_noise_level, size=x_context.shape)
        if self.context_apply_to == "y" or self.context_apply_to == "both":
            y_context += torch.normal(0, self.context_noise_level, size=y_context.shape)
        return x_context, y_context

    def add_noise_target(self, x_target, y_target) -> \
            Tuple[Float[Tensor, "batch target in"], Float[Tensor, "batch target out"],]:
        """
        Adds homogeneous gaussian noise to target data
        Noise can be applied to x, y or both
        Args:
            x_context: shape (batch_size, n_context, input_dim)
            y_context: shape (batch_size, n_context, output_dim)

        Returns:

        """
        if self.target_apply_to == "x" or self.target_apply_to == "both":
            x_target += torch.normal(0., self.target_noise_level, size=x_target.shape)
        if self.target_apply_to == "y" or self.target_apply_to == "both":
            y_target += torch.normal(0., self.target_noise_level, size=y_target.shape)
        return x_target, y_target
