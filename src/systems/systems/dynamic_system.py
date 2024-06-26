from dataclasses import dataclass
from typing import Union, List, Dict, Tuple, Callable, Mapping, Sequence

import numpy as np
import torch
from jaxtyping import Float
from omegaconf import ListConfig, DictConfig
from torch import Tensor
from torch.types import Device
from typeguard import typechecked


@dataclass
class DynamicSystemType:
    state_size: int
    control_size: int
    state_names: Tuple[str]
    control_names: Tuple[str]
    cost_names: Tuple[str]
    lower_control_bounds: Tuple[float]
    upper_control_bounds: Tuple[float]
    inc_std: Tuple[int]


@dataclass
class VanDerPolType(DynamicSystemType):
    state_size: int = 2
    control_size: int = 1
    state_names: Tuple[str] = ("x", "y")
    control_names: Tuple[str] = ("u",)
    cost_names: Tuple[str] = ("distance", "speed")
    lower_control_bounds: Tuple[float] = (-4,)
    upper_control_bounds: Tuple[float] = (4,)
    inc_std: Tuple[int] = (0.02093719, 0.15849732)


@dataclass
class MultiModalVanDerPolType(DynamicSystemType):
    state_size: int = 2
    control_size: int = 1
    state_names: Tuple[str, str] = ("x", "y")
    control_names: Tuple[str] = ("u",)
    cost_names: Tuple[str, str] = ("distance", "speed")
    lower_control_bounds: Tuple[float] = (-1,)
    upper_control_bounds: Tuple[float] = (1,)
    inc_std: Tuple[int] = None

@dataclass
class HalfCheethaType(DynamicSystemType):
    state_size: int = 18
    control_size: int = 6
    state_names: Tuple[str, str] = ("rootx", "rootz", "roota", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot",
                                    "rootxv", "rootzv", "rootav", "bthigh_av", "bshin_av", "bfoot_av", "fthigh_av",
                                    "fshin_av", "ffoot_av")
    control_names: Tuple[str] = ("bthigh_u", "bshin_u", "bfoot_u", "fthigh_u", "fshin_u", "ffoot_u")
    cost_names: Tuple[str] = ("x_speed",)
    lower_control_bounds: Tuple[str] = (-1,)
    upper_control_bounds: Tuple[str] = (1,)
    inc_std: Tuple[int] = (0.1306,  0.0661,  0.1345,  0.4531,  0.4728,  0.3633,  0.4327,  0.4152,
         0.2128,  1.4575,  1.4809,  3.2798, 13.6731, 16.3548, 13.6335, 12.8149,
        13.1521,  6.7177)

@dataclass
class HopperType(DynamicSystemType):
    state_size: int = 12
    control_size: int = 3
    state_names: Tuple[str, str] = ("rootx", "rootz", "roota", "thigh", "leg", "foot", "xv", "zv", "av", "thigh_av",
                                    "leg_av", "foot_av")
    control_names: Tuple[str] = ("thigh_u", "leg_u", "foot_u")
    cost_names: Tuple[str] = ("x_speed",)
    lower_control_bounds: Tuple[str] = (-1,)
    upper_control_bounds: Tuple[str] = (1,)
    inc_std: Tuple[int] = (0.2796,  0.0535,  0.1151,  0.3957,  0.4147,  0.2801,  0.3710,  0.3489,
         0.1809,  1.2628,  1.2090,  2.8765, 11.4672, 14.5356, 10.3542, 10.6428,
        10.6844,  5.6218)

@dataclass
class AntType(DynamicSystemType):
    state_size: int = 29
    control_size: int = 8
    state_names: Tuple[str, str] = ("rootx",
                                    "rooty",
                                    "rootz",
                                    "rootxa",
                                    "rootya",
                                    "rootza",
                                    "rootwa",
                                    "hip_1",
                                    "ankle_1",
                                    "hip_2",
                                    "ankle_2",
                                    "hip_3",
                                    "ankle_3",
                                    "hip_4",
                                    "ankle_4",
                                    "torso_xv",
                                    "torso_yv",
                                    "torso_zv",
                                    "torso_xav",
                                    "torso_yav",
                                    "torso_zav",
                                    "hip_1_av",
                                    "ankle_1_av",
                                    "hip_2_av",
                                    "ankle_2_av",
                                    "hip_3_av",
                                    "ankle_3_av",
                                    "hip_4_av",
                                    "ankle_4_av")

    control_names: Tuple[str] = ("hip_4_u", "ankle_4_u", "hip_1_u", "ankle_1_u", "hip_2_u", "ankle_2_u", "hip_3_u", "ankle_3_u")
    cost_names: Tuple[str] = ("total"),
    lower_control_bounds: Tuple[str] = (-1,)
    upper_control_bounds: Tuple[str] = (1,)
    inc_std: Tuple[int] = None

@dataclass
class InvertedPendulumType(DynamicSystemType):
    state_size: int = 4
    control_size: int = 1
    state_names: Tuple[str] = ("x", "xd", "theta", "thetad")
    control_names: Tuple[str] = ("u",)
    cost_names: Tuple[str] = ("control", "state")
    lower_control_bounds: Tuple[str] = (-3,)
    upper_control_bounds: Tuple[str] = (3,)
    inc_std: Tuple[int] = (0.02960792, 0.09526347, 0.11824167, 0.3855852)


class DynamicSystem:
    """
    Dynamic system class
    Provides methods to predict the next state and the trajectory with or without the initial state
    """

    def __init__(self,
                 delta_t: float = 0.02,
                 device: str = "cpu",
                 control_size: int = None,
                 state_size: int = None,
                 state_names: Tuple[str] = None,
                 control_names: Tuple[str] = None,
                 system_type: DynamicSystemType = None):
        assert (
                       control_size is None and state_size is None) or system_type is None, "Either specify control_size and state_size or system_type"
        self.state_size = state_size
        self.control_size = control_size
        self.state_names = state_names
        self.control_names = control_names
        self.example_init_states_ranges = None
        self.example_parameter_ranges = None
        self.delta_t = delta_t
        if system_type is not None:
            self._init_vars_with_type(system_type)
        self.device = torch.device(device)

    def _init_vars_with_type(self, system_type: DynamicSystemType):
        self.state_size = system_type.state_size
        self.control_size = system_type.control_size
        self.state_names = system_type.state_names
        self.control_names = system_type.control_names

    def description(self) -> Dict[str, Union[str, float]]:
        """
        Returns a description of the system including the name and the relevant parameters
        Returns:
            description: description of the system as a dictionary
        """
        return {
            "class": self.__class__.__name__,
            "delta_t": self.delta_t
        }

    def to(self, device: Device):
        self.device = device

    def next_state(self,
                   state: Float[Tensor, "batch state"],
                   control: Float[Tensor, "batch control"] = None) -> \
            Float[torch.Tensor, "batch state"]:
        """
        Predicts the next state given the current state and the control input
        Args:
            state: state; shape(batch_size, state_size)
            control: control input; shape(batch_size, control_size)

        Returns:
            next_state: next state; shape(batch_size, state_size)
        """
        raise NotImplemented()

    def predict_traj(self,
                     initial_state: Float[Tensor, "batch state"],
                     control: Float[Tensor, "batch time control"]) -> \
            Float[Tensor, "batch time state"]:
        """
        Predicts the trajectory given the initial state and the control input
        Does not include the initial state
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)

        Returns:
            trajectory: trajectory; shape(batch_size, time, state_size)
        """
        return self.predict_traj_with_init(initial_state, control)[:, 1:, :]

    def predict_traj_with_init(self,
                               initial_state: Float[Tensor, "batch state"],
                               control: Float[Tensor, "batch time control"] = None, steps: int = None) -> \
            Float[Tensor, "batch time+1 state"]:
        """
        Predicts the trajectory given the initial state and the control input
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size)

        Returns:
            trajectory: trajectory; shape(batch_size, time+1, state_size)
        """
        assert (control is not None) != (steps is not None), "Give either control or steps"
        if steps is not None:
            control = torch.zeros((initial_state.shape[0], steps, self.control_size))
        s = torch.empty((initial_state.shape[0], control.shape[1] + 1, initial_state.shape[1]), device=self.device)
        s[:, 0, :] = initial_state
        for t in range(control.shape[-2]):
            s_new = self.next_state(s[:, t], control[:, t])
            s[:, t + 1] = s_new
        return s


class ExactDynamicSystem(DynamicSystem):
    """
    Dynamic system for which the next state can be calculated exactly with a mathematical formula
    """

    def __init__(self, parameter_names: Sequence[str] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parameter_names = parameter_names

    def get_parameters(self) -> Dict[str, float]:
        """
        Returns the parameters of the system
        Returns:
            parameters: parameters of the system as a dictionary
        """
        raise NotImplemented()

    def derivation(self,
                   state: Float[Tensor, "batch state"],
                   control: Float[Tensor, "batch control"]) -> \
            Float[Tensor, "*time state"]:
        raise NotImplemented()

    def to(self, device: Device):
        super().to(device)
        # Call to for all parameters in self.parameter_names
        for parameter_name in self.parameter_names:
            parameter = getattr(self, parameter_name)
            if isinstance(parameter, torch.Tensor):
                setattr(self, parameter_name, parameter.to(device))

    def next_state(self,
                   state: Float[Tensor, "batch state"],
                   control: Float[Tensor, "batch control"] = None,
                   method: str = "rk4") -> \
            Float[Tensor, "batch state"]:
        """
        Predicts the next state given the current state and the control input
        Uses the rk4 method
        Args:
            state: state; shape(batch_size, state_size)
            control: control input; shape(batch_size, control_size)
            method: Method to calculate the next step. Either "rk4" or "euler"

        Returns:
            next_state: next state; shape(batch_size, state_size)
        """
        if method == "euler":
            return self.next_state_euler(state, control)
        elif method == "rk4":
            return self.next_state_rk4(state, control)

    def predict_traj_with_init(self,
                               initial_state: Float[Tensor, "batch state"],
                               control: Float[Tensor, "batch time control"] = None,
                               time: float = None,
                               steps: int = None,
                               method: str = "rk4",
                               parameter_func: Callable[[float], Dict[str, float]] = None) \
            -> Float[Tensor, "batch time+1 state"]:
        """
        Predicts the trajectory given the initial state (Extended interface in relation to the base class)
        One can give a control input or just a prediction time. In the latter case, the control input is set to zero
        If prediction time is given, either the number of steps or the prediction time must be given
        Args:
            initial_state: initial state; shape(batch_size, state_size)
            control: control input; shape(batch_size, time, control_size) (Optional)
            time: Length of prediction in seconds (Optional)
            steps: Length of prediction in steps (Optional)
            method: Method to calculate the next step. Either "rk4" or "euler"
            parameter_func: Function that can be used to change the parameters of the system during the prediction
                The function must take the time in second as an argument and return the new parameters Dict[str, float]

        Returns:
            trajectory: trajectory; shape(batch_size, time+1, state_size)
        """
        assert method in ["rk4", "euler"]
        if time:
            steps = int(time / self.delta_t)
        if control is None:
            assert (steps is None) != (time is None), "Give either steps or time"
            control = torch.zeros((initial_state.shape[0], steps, self.control_size), device=self.device)
        else:
            assert steps is None and time is None, "Give either control or steps/time"
        s = torch.empty((initial_state.shape[0], control.shape[1] + 1, initial_state.shape[1]), device=self.device)
        s[:, 0, :] = initial_state
        for t in range(control.shape[-2]):
            if parameter_func is not None:
                self.set_parameter(parameter_func(t * self.delta_t))
            s_new = self.next_state(s[:, t], control[:, t])
            s[:, t + 1] = s_new
        return s

    def sample_init_state(self,
                          ranges: Union[Sequence[Sequence[float]]] = None,
                          number: int = 1) \
            -> Float[torch.Tensor, "batch states"]:
        """
        Samples initial states from the given ranges
        Args:
            ranges: ranges of the initial states; shape(state_size, 2)
            number: number of initial states to sample

        Returns:
            states: sampled initial states; shape(number, state_size)
        """
        random_state_values = []
        if not ranges:
            ranges = self.example_init_states_ranges
        for i in range(len(ranges)):
            random_values = torch.distributions.uniform.Uniform(ranges[i][0],
                                                                ranges[i][1]) \
                .sample(torch.Size([number])).to(self.device)
            random_state_values.append(random_values)
        random_state_values = torch.stack(random_state_values).permute((1, 0))
        return random_state_values

    def sample_system_parameter(self, ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]] = None) \
            -> Dict[str, Float[Tensor, "1"]]:
        """
        Samples system parameters from the given ranges
        Args:
            ranges: ranges of the system parameters; Dict[parameter_name, List[lower_bound, upper_bound]]

        Returns:
            parameters: sampled system parameters; Dict[parameter_name, value]
        """
        parameters = {}
        if not ranges:
            ranges = self.example_parameter_ranges
        for attr, att_range in ranges.items():
            if isinstance(att_range, List) or isinstance(att_range, ListConfig):
                value = torch.distributions.uniform.Uniform(att_range[0], att_range[1]).sample([1]).to(
                    self.device)
            else:
                value = torch.tensor(att_range).to(self.device)
            parameters[attr] = value
        return parameters

    def set_parameter(self, parameters: Union[
        Mapping["str", float],
        List[float],
        List[Float[Tensor, "batch"]],
        Float[Tensor, "batch state"],
        Mapping["str", Float[Tensor, "batch"]]]):
        """
        Applies the given parameters to the system
        If its a list of parameters, the order must be the same as the order of the parameters in the system
        Args:
            parameters: Dict with parameter names as keys and parameter values as values
                or list of parameter values
        """
        if isinstance(parameters, Mapping):
            for attr, value in parameters.items():
                setattr(self, attr, value)
        elif isinstance(parameters, List):
            for i, value in enumerate(parameters):
                setattr(self, self.parameter_names[i], value)
        elif isinstance(parameters, Tensor):
            for i in range(parameters.size(1)):
                setattr(self, self.parameter_names[i], parameters[:, i])

    def next_state_euler(self,
                         state: Float[Tensor, "batch state"],
                         control: Float[Tensor, "batch control"] = None) -> \
            Float[Tensor, "batch state"]:
        """
        Predicts the next state given the current state and the control input
        Uses the euler method
        Args:
            state: state; shape(batch_size, state_size)
            control: control input; shape(batch_size, control_size)

        Returns:
            next_state: next state; shape(batch_size, state_size)
        """
        if not control:
            control = torch.zeros((state.shape[0], self.control_size), device=self.device)
        return state + self.derivation(state, control) * self.delta_t

    def next_state_rk4(self,
                       state: Float[Tensor, "*batch state"],
                       control: Float[Tensor, "*batch control"] = None) -> \
            Float[Tensor, "*batch state"]:
        """
        Predicts the next state given the current state and the control input
        Uses the rk4 method
        Args:
            state: state; shape(batch_size, state_size)
            control: control input; shape(batch_size, control_size)

        Returns:
            next_state: next state; shape(batch_size, state_size)
        """
        if control is None:
            control = torch.zeros((state.shape[0], self.control_size), device=self.device)
        k1 = self.derivation(state, control)
        k2 = self.derivation(state + self.delta_t * k1 / 2, control)
        k3 = self.derivation(state + self.delta_t * k2 / 2, control)
        k4 = self.derivation(state + self.delta_t * k3, control)
        return state + (1 / 6) * (k1 + 2 * k2 + 2 * k3 + k4) * self.delta_t

    def clip_parameters(self, init_sample_ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]]):
        """
        Clips the parameters of the system to the given ranges
        Args:
            init_sample_ranges:

        Returns:

        """
        for attr, att_range in init_sample_ranges.items():
            if isinstance(att_range, Sequence):
                value = np.clip(getattr(self, attr), att_range[0], att_range[1])
                setattr(self, attr, value)
            else:
                value = np.clip(getattr(self, attr), att_range, att_range)
                setattr(self, attr, value)

    def get_parameters_as_list(self):
        """
        Returns the parameters of the system as a list
        Returns:

        """
        return [getattr(self, attr) for attr in self.parameter_names]
