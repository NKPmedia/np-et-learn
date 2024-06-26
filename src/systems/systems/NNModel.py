from typing import Tuple, Sequence

import torch
from jaxtyping import Float
from torch import Tensor
from torch.distributions import Distribution
from torch.types import Device

from src.modules.base_np import BaseNP
from src.modules.cnp import CNP
from src.modules.nd_nn import NdNN
from src.systems.systems.dynamic_system import DynamicSystem, DynamicSystemType


class NNModel(DynamicSystem):
    """
    Dynamic system that draws is dynamics from a neural network
    Cant compute a derivation
    """

    def __init__(self,
                 model_path: str = "",
                 model: BaseNP.__class__ = None,
                 loaded_model: BaseNP = None,
                 state_size: int = None,
                 control_size: int = None,
                 system_type: DynamicSystemType = None,
                 device: str = "cpu",
                 strict_load: bool = True,
                 compile: str = "none",
                 compile_mode: str = "default",
                 use_latent_mean: bool = True,
                 num_z_samples: int = 1):
        super().__init__(device=device, state_size=state_size, control_size=control_size, system_type=system_type)
        assert compile in ["none", "torch-trace", "dynamo"], "Compile must be none or torch-trace"
        assert (model_path != "") != (loaded_model is not None), "Give either model_path or model"
        assert compile_mode in ["reduce-overhead", "default", "max-autotune"], \
            "Compile mode must be reduce-overhead, default or max-autotune"
        if model_path != "":
            self.nn_model = model.load_from_checkpoint(model_path, strict=strict_load, map_location=torch.device("cpu"))
        else:
            self.nn_model = loaded_model

        self.compile_mode = compile_mode
        self.use_latent_mean = use_latent_mean
        self.num_z_samples = num_z_samples

        self.to(self.device)
        self.nn_model.eval()
        self.compile(compile)

        if isinstance(self.nn_model, CNP):
            self.model_type = "cnp"
        elif isinstance(self.nn_model, NdNN):
            self.model_type = "base-nn"
        else:
            try:
                self.model_type = self.nn_model.model_type
            except AttributeError:
                raise ValueError("Model type could not be determined")

    def compile(self, compile: str):
        if compile == "none":
            self.forward = self.nn_model.forward
            self.unroll_forward = self.nn_model.unroll_forward
        elif compile == "torch-trace":
            print("Compiling model to torch-trace")
            self.nn_model_script = self.nn_model.to_torchscript(method="trace")
            self.forward = self.nn_model_script.forward
            self.unroll_forward = self.nn_model.unroll_forward
        elif compile == "dynamo":
            print("Compiling model to dynamo")
            #raise NotImplementedError("Dynamo causes probles at the moment. "
            #                          "We have to implement recompilation or check if the model is correctly adapted")
            self.forward = torch.compile(self.nn_model.forward, mode=self.compile_mode, backend="inductor")
            self.forward_unoptimized = self.nn_model.forward
            self.unroll_forward = torch.compile(self.nn_model.unroll_forward, mode=self.compile_mode, backend="inductor")
            self.unroll_forward_unoptimized = self.nn_model.unroll_forward
        pass

    def to(self, device: Device):
        super().to(device)
        self.nn_model.to(device)

    def predict_traj_with_init(self,
                               initial_state: Float[Tensor, "batch state"],
                               control: Float[Tensor, "batch time control"] = None,
                               steps: int = None,
                               unroll_mode: str = "mean_propagation") -> \
            Float[Tensor, "batch time+1 state"]:
        assert (control is not None) != (steps is not None), "Give either control or steps"
        with torch.no_grad():
            if steps is not None:
                control = torch.zeros((initial_state.shape[0], steps, self.control_size))

            s_dist, _, _, _ = self.unroll_forward(self.context_x,
                                                  self.context_y,
                                                  initial_state[:, None, :],
                                                  control,
                                                  use_latent_mean=self.use_latent_mean,
                                                  num_z_samples=self.num_z_samples,
                                                  unroll_mode=unroll_mode)
        return s_dist.mean

    def predict_mean_std_with_init(self,
                                   initial_state: Float[Tensor, "batch state"],
                                   control: Float[Tensor, "batch time control"] = None,
                                   steps: int = None,
                                   context_sizes: Sequence[int] = None,
                                   pop_size: int = 1,
                                   unroll_mode: str = "mean_propagation") -> \
            Tuple[Float[Tensor, "batch time+1 state"], Float[Tensor, "batch time+1 state"]]:
        """
        Predicts the trajectory given the initial state and the control input
        Returns the mean and plot_std given by the neural network
        The trajectory does include the initial state
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)
            steps: number of steps to predict

        Returns:
            s: trajectory mean; shape(batch_size, time+1, state_size)
            plot_std: trajectory plot_std; shape(batch_size, time+1, state_size)
        """
        assert context_sizes is not None, "Context sizes must be given"
        with torch.no_grad():
            assert (control is not None) != (steps is not None), "Give either control or steps"
            if steps is not None:
                control = torch.zeros((initial_state.shape[0], steps, self.control_size))
            t_dist, _, _, _ = self.unroll_forward(self.context_x,
                                                  self.context_y,
                                                  initial_state[:, None, :],
                                                  control,
                                                  context_sizes=context_sizes,
                                                  pop_size=pop_size,
                                                  use_latent_mean=self.use_latent_mean,
                                                  num_z_samples=self.num_z_samples,
                                                  unroll_mode=unroll_mode)
        return t_dist.mean, t_dist.stddev

    def predict_mean_std_z_with_init(self,
                                   initial_state: Float[Tensor, "batch state"],
                                   control: Float[Tensor, "batch time control"] = None,
                                   steps: int = None,
                                   context_sizes: Sequence[int] = None,
                                   pop_size: int = 1,
                                   unroll_mode: str = "mean_propagation") -> \
            Tuple[Float[Tensor, "batch time+1 state"], Float[Tensor, "batch time+1 state"]
            , Float[Tensor, "batch latent"], Distribution]:
        """
        Predicts the trajectory given the initial state and the control input
        Returns the mean and plot_std given by the neural network
        The trajectory does include the initial state
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)
            steps: number of steps to predict

        Returns:
            s: trajectory mean; shape(batch_size, time+1, state_size)
            plot_std: trajectory plot_std; shape(batch_size, time+1, state_size)
        """
        with torch.no_grad():
            assert (control is not None) != (steps is not None), "Give either control or steps"
            if steps is not None:
                control = torch.zeros((initial_state.shape[0], steps, self.control_size))
            t_dist, z_latent, z_dist, _ = self.unroll_forward(self.context_x,
                                                  self.context_y,
                                                  initial_state[:, None, :],
                                                  control,
                                                  context_sizes=context_sizes,
                                                  pop_size=pop_size,
                                                  use_latent_mean=self.use_latent_mean,
                                                  num_z_samples=self.num_z_samples,
                                                  unroll_mode=unroll_mode)
        return t_dist.mean, t_dist.stddev, z_latent, z_dist

    def predict_mean_std(self,
                         initial_state: Float[Tensor, "batch state"],
                         control: Float[Tensor, "batch time control"] = None,
                         steps: int = None,
                         context_sizes: Sequence[int] = None,
                         pop_size: int = 1,
                         unroll_mode: str = "mean_propagation") -> \
            Tuple[Float[Tensor, "batch time state"], Float[Tensor, "batch time state"]]:
        """
        Predicts the trajectory given the initial state and the control input
        Returns the mean and plot_std given by the neural network
        The trajectory does not include the initial state
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)
            steps: number of steps to predict

        Returns:
            s: trajectory mean; shape(batch_size, time, state_size)
            plot_std: trajectory plot_std; shape(batch_size, time, state_size)
        """
        mean, std = self.predict_mean_std_with_init(initial_state, control, steps, context_sizes, pop_size, unroll_mode)
        return mean[:, 1:], std[:, 1:]

    def predict_mean_std_z(self,
                         initial_state: Float[Tensor, "batch state"],
                         control: Float[Tensor, "batch time control"] = None,
                         steps: int = None,
                         context_sizes: Sequence[int] = None,
                         pop_size: int = 1,
                         unroll_mode: str = "mean_propagation") -> \
            Tuple[Float[Tensor, "batch time state"], Float[Tensor, "batch time state"],
            Float[Tensor, "batch latent"], Distribution]:
        """
        Predicts the trajectory given the initial state and the control input
        Returns the mean and plot_std given by the neural network
        The trajectory does not include the initial state
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)
            steps: number of steps to predict

        Returns:
            s: trajectory mean; shape(batch_size, time, state_size)
            plot_std: trajectory plot_std; shape(batch_size, time, state_size)
        """
        mean, std, z_latent, z_dist = self.predict_mean_std_z_with_init(initial_state, control, steps, context_sizes, pop_size, unroll_mode)
        return mean[:, 1:], std[:, 1:], z_latent, z_dist

    def set_context(self,
                    context_x: Float[torch.Tensor, "batch context state_control"],
                    context_y: Float[torch.Tensor, "batch context state"]):
        """
        Sets the context of the neural network (Conditional neural networks)
        Its basically the input and output of the next_state function of a real dynamic system
        The contexts a re not shares between the batches
        Args:
            context_x: states of the system; shape(batch_size, context_size, state_size)
            context_y: next states of the system; shape(batch_size, context_size, state_size)
        """
        self.context_x = context_x
        self.context_y = context_y

    def next_state(self,
                   state: Float[Tensor, "batch target state"],
                   control: Float[Tensor, "batch target control"] = None) -> \
            Float[torch.Tensor, "batch target state"]:
        with torch.no_grad():
            if control is None:
                control = torch.zeros((state.size(0), state.size(1), self.control_size), device=state.device)
            s_dist, _, _, _ = self.forward(self.context_x,
                                           self.context_y,
                                           torch.concatenate([state, control], dim=2),
                                           use_latent_mean=True)
            if self.model_type == "lnp":
                return s_dist.mean[:, 0]
        return s_dist.mean

    def next_state_with_std(self,
                            state: Float[Tensor, "batch target state"],
                            control: Float[Tensor, "batch target control"] = None,
                            context_sizes: Sequence[int] = None) -> \
            Tuple[Float[torch.Tensor, "batch target state"], Float[torch.Tensor, "batch target state"]]:
        """
        Predicts the next states mean and plot_std given the current state and the control input
        Args:
            context_sizes: sizes of the context to use for each batch (can be used if the context is padded)
            state: current state; shape(batch_size, target, state_size)
            control: control input; shape(batch_size, target, control_size)

        Returns:
            s: next state mean; shape(batch_size, target, state_size)
            std: next state std; shape(batch_size, target, state_size)
        """
        with torch.no_grad():
            if control is None:
                control = torch.zeros((state.size(0), state.size(1), self.control_size), device=state.device)
            s_dist, _, _, _ = self.forward(self.context_x,
                                           self.context_y,
                                           torch.concatenate([state, control], dim=2),
                                           context_sizes=context_sizes,
                                           use_latent_mean=True)
            if self.model_type == "lnp":
                return s_dist.mean[:, 0], s_dist.stddev[:, 0]
        return s_dist.mean, s_dist.stddev

    def get_latent_std(self, state: Float[Tensor, "batch target state"],
                       control: Float[Tensor, "batch target control"] = None,
                       context_sizes: Sequence[int] = None) -> \
            Float[torch.Tensor, "batch hidden_state"]:
        """
        Returns the latent space standard deviation given the context, current state and the control input
        Args:
            state: current state; shape(batch_size, target, state_size)
            control: control input; shape(batch_size, target, control_size)
            context_sizes: sizes of the context to use for each batch (can be used if the context is padded)

        Returns:
            std: latent space standard deviation; shape(batch_size, hidden_size)

        """
        _, z_sample, z_dist, _ = self.forward(self.context_x,
                                              self.context_y,
                                              torch.concatenate([state, control], dim=2),
                                              context_sizes=context_sizes,
                                              use_latent_mean=True)
        if z_dist is None:
            assert self.model_type == "cnp" or self.model_type == "base-nn"
            return torch.zeros((state.size(0), self.nn_model.z_latent_dim), device=state.device)
        return z_dist.stddev
