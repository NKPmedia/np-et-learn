import abc
from numbers import Number
from typing import Union, List, Sequence, Mapping

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor


class ParameterChangeable(abc.ABC):

    @abc.abstractmethod
    def get_parameter_names(self) -> List[str]:
        raise NotImplemented()

    @abc.abstractmethod
    def set_parameter(self, parameter: Float[Tensor, "batch parameter"]):
        raise NotImplemented()

    @abc.abstractmethod
    def get_parameters(self):
        raise NotImplemented()

    @abc.abstractmethod
    def clip_parameters(self, init_sample_ranges):
        raise NotImplemented()


class ParameterChanger:

    def __init__(self):
        self.init_sample_ranges = None
        self.current_parameter = {}

    def set_init_sample_ranges(self, init_sample_ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]]):
        self.init_sample_ranges = init_sample_ranges

    def sample_system_parameter(self,
                                sample_ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]],
                                model: ParameterChangeable,
                                batch_size: int = 1) \
            -> Float[Tensor, "batch parameter"]:
        parameter_names = model.get_parameter_names()
        parameters = torch.zeros((batch_size, len(parameter_names)))
        for batch in range(batch_size):
            for i, prameter_name in enumerate(parameter_names):
                if prameter_name not in sample_ranges:
                    print(f"Parameter {prameter_name} not in sample_ranges, using default value")
                    continue
                val_range = sample_ranges[prameter_name]
                if isinstance(val_range, Sequence):
                    assert len(val_range) > 0, "val_range must have at least one element"
                    # Select random list from the lists
                    if isinstance(val_range[0], Sequence):
                        val_range = val_range[
                            torch.randint(0, len(val_range), (1,), generator=self.generator).to(torch.int).item()
                        ]
                    # Select random value from the list or sample from uniform distribution if two values are given
                    if len(val_range) == 2:

                        sampled_value = (torch.rand((1,), generator=self.generator).item()
                                         * (val_range[1] - val_range[0])) + val_range[0]
                    else:
                        sampled_value = val_range[
                            torch.randint(0, len(val_range), (1,), generator=self.generator).to(torch.int).item()
                        ]
                else:
                    # Take single value
                    sampled_value = torch.tensor([val_range])
                parameters[batch, i] = sampled_value
        return parameters

    def reset(self, model: ParameterChangeable, batch_size: int = 1, seed: int = None):
        """
        Resets the models parameters to the initial values (Can be random or fixed)
        Returns:

        """
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        parameter = self.sample_system_parameter(self.init_sample_ranges, model, batch_size)
        model.set_parameter(parameter)
        self.last_parameter = parameter

    def parameter_step(self, system: ParameterChangeable, time_step: int):
        """
        Changes the parameters of the system
        One step is one step in the system simulation
        Args:
            time_step:
            system:

        Returns:

        """
        raise NotImplemented()


class MaxSetAwareParameterChanger(ParameterChanger):
    def __init__(self, max_set_size: int = -1):
        super().__init__()
        self.last_parameter_set_nrs = None
        self.max_set_size = max_set_size
        self.parameter_set = []
        self.reuse_parameter = False
        self.last_parameter = None

    def get_last_parameter_set_nrs(self):
        assert self.last_parameter_set_nrs is not None, "reset first to get first parameter set nr"
        return self.last_parameter_set_nrs

    def toggle_reuse_parameter_flag(self, reuse_parameter: bool):
        self.reuse_parameter = reuse_parameter

    def reset(self, model: ParameterChangeable, batch_size: int = 1, seed: int = None):
        """
        Resets the models parameters to the initial values (Can be random or fixed)
        Returns:

        """
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

        if (len(self.parameter_set) >= self.max_set_size != -1) or self.reuse_parameter:
            parameter = []
            idxs = []
            for i in range(batch_size):
                idx = torch.randint(0, len(self.parameter_set), (1,), generator=self.generator).item()
                parameter.append(self.parameter_set[idx])
                idxs.append(idx)
            parameter = torch.stack(parameter, dim=0)
            model.set_parameter(parameter)
            self.last_parameter_set_nrs = idxs
        else:
            parameter = self.sample_system_parameter(self.init_sample_ranges, model, batch_size)

            self.last_parameter_set_nrs = self._add_parameter_to_set(parameter)
            model.set_parameter(parameter)
        self.last_parameter = parameter

    def _add_parameter_to_set(self, parameter: Float[Tensor, "batch parameter"]) -> List[int]:
        # check ig parameter is already in set
        idx = []
        for new_parameter in parameter:
            is_in_set = False
            for i, old_param in enumerate(self.parameter_set):
                if torch.allclose(old_param, new_parameter):
                    idx.append(i)
                    is_in_set = True
                    break
            # if not add it
            if not is_in_set:
                self.parameter_set.append(new_parameter)
                idx.append(len(self.parameter_set) - 1)
        return idx


class ConstantParameter(MaxSetAwareParameterChanger):
    """
    Does not change the parameters of the system after the initialization
    """

    def parameter_step(self, system: ParameterChangeable, time_step: int):
        """
        Does not change the parameters
        Args:
            system:

        Returns:

        """
        if self.last_parameter is not None:
            system.set_parameter(self.last_parameter)
        else:
            pass


class SingleRandomParameterChange(ParameterChanger):
    """
    Does change the parameters of the system after the initialization at a random time step with a random amount
    """

    def __init__(self,
                 change_step_range: Sequence[int],
                 change_value_range: Sequence[float],
                 change_parameters: Sequence[str],
                 parameter_warparound: bool = False):
        """
        change_step will be the index of the time step when the parameter will be changed
        A change at x will bew visible at x+1 (counting: init state is state 0)
        Args:
            change_step_range:
            change_percentage_range:
            change_parameters:
        """
        super().__init__()
        self.change_parameters = change_parameters
        self.change_step_range = change_step_range
        self.change_value_range = change_value_range
        self.parameter_warparound = parameter_warparound
        self.step = 0

    def reset(self, model: ParameterChangeable, batch_size: int = 1, seed: int = None):
        """
        Resets the models parameters to the initial values (Can be random or fixed)
        Returns:

        """
        assert batch_size == 1, "SingleRandomParameterChange only supports batch_size = 1"
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None
        parameter = self.sample_system_parameter(self.init_sample_ranges, model)
        model.set_parameter(parameter)
        self.step = 0
        self.change_step = torch.randint(self.change_step_range[0], self.change_step_range[1], (1,)).item()
        self.change_value = torch.rand(1).item() * (self.change_value_range[1] - self.change_value_range[
            0]) + self.change_value_range[0]

    def parameter_step(self, system: ParameterChangeable, time_step: int):
        """
        Does change the parameters at the set time step by a set percentage
        Args:
            time_step:
            system:

        Returns:

        """
        assert time_step == self.step, "Time step is not the same as the step counter, programming error?"
        if self.step == self.change_step:
            current_parameter = system.get_parameters()
            assert isinstance(current_parameter, Mapping), "current_parameter must be a mapping"
            for parameter in self.change_parameters:
                current_parameter[parameter] += self.change_value * np.random.choice([-1, 1], 1).item()
            current_parameter = clip_parameters(current_parameter, self.init_sample_ranges,
                                                system.get_parameter_names(), self.parameter_warparound)
            system.set_parameter(current_parameter)
            system.clip_parameters(self.init_sample_ranges)

        self.step += 1


def clip_parameters(current_parameter: Float[Tensor, "batch parameter"],
                    init_sample_ranges: Mapping[str, Sequence[float]],
                    parameter_names: Sequence[str],
                    parameter_warparound: bool = False):
    # check the type of init_sample_ranges
    is_mapping = False
    if isinstance(current_parameter, Mapping):
        is_mapping = True
        current_parameter = [current_parameter[parameter] for parameter in parameter_names]
        current_parameter = torch.stack(current_parameter, dim=1)
    if current_parameter.ndim == 1:
        current_parameter = current_parameter.unsqueeze(0)
    assert isinstance(init_sample_ranges, Mapping), "init_sample_ranges must be a mapping"
    for i, parameter in enumerate(parameter_names):
        if parameter in init_sample_ranges:
            if isinstance(init_sample_ranges[parameter], Number):
                current_parameter[:, i] = init_sample_ranges[parameter]
            else:
                assert isinstance(init_sample_ranges[parameter], Sequence), \
                    f"init_sample_ranges[{parameter}] must be a sequence"
                if parameter_warparound:
                    clamped_parameter = torch.clamp(current_parameter[:, i],
                                                    init_sample_ranges[parameter][0],
                                                    init_sample_ranges[parameter][1])
                    overshoot = current_parameter[:, i] - clamped_parameter
                    current_parameter[:, i] = clamped_parameter - overshoot
                else:
                    current_parameter[:, i] = torch.clamp(current_parameter[:, i],
                                                          init_sample_ranges[parameter][0],
                                                          init_sample_ranges[parameter][1])

        else:
            raise ValueError(f"Parameter {parameter} not in init_sample_ranges")
    if is_mapping:
        return {parameter_names[i]: current_parameter[:, i] for i in range(len(parameter_names))}
    return current_parameter


class SingleRandomMultiParameterChange(SingleRandomParameterChange):
    """
    Does change the parameters of the system after the initialization at a random time step with a random amount
    """

    # noinspection PyMissingConstructor
    def __init__(self,
                 change_step_range: Sequence[int],
                 change_value_range: Sequence[Sequence[float]],
                 change_parameters: Sequence[str],
                 parameter_warparound: bool = False):
        """
        change_step will be the index of the time step when the parameter will be changed
        A change at x will bew visible at x+1 (counting: init state is state 0)
        Args:
            change_step_range:
            change_percentage_range:
            change_parameters:
        """
        self.change_parameters = change_parameters
        self.change_step_range = change_step_range
        self.change_value_range = change_value_range
        self.parameter_warparound = parameter_warparound
        self.step = 0
        self.init_sample_ranges = None
        self.current_parameter = {}

    def reset(self, model: ParameterChangeable, batch_size: int = 1, seed: int = None):
        """
        Resets the models parameters to the initial values (Can be random or fixed)
        Returns:

        """
        assert batch_size == 1, "SingleRandomParameterChange only supports batch_size = 1"
        if seed is not None:
            self.generator = torch.Generator()
            self.generator.manual_seed(seed)
        else:
            self.generator = None

        parameter = self.sample_system_parameter(self.init_sample_ranges, model)
        model.set_parameter(parameter)
        self.step = 0
        self.change_step = torch.randint(self.change_step_range[0], self.change_step_range[1], (1,)).item()
        self.change_value = []
        for i in range(len(self.change_parameters)):
            self.change_value.append(
                torch.rand(1).item() * (self.change_value_range[i][1] - self.change_value_range[i][0]) +
                self.change_value_range[i][0])

    def parameter_step(self, system: ParameterChangeable, time_step: int):
        """
        Does change the parameters at the set time step by a set percentage
        Args:
            time_step:
            system:

        Returns:

        """
        assert time_step == self.step, "Time step is not the same as the step counter, programming error?"
        if self.step == self.change_step:
            print("change parameters")
            current_parameter = system.get_parameters().clone()
            assert isinstance(current_parameter, Tensor), "current_parameter must be a tensor"
            for parameter in self.change_parameters:
                if isinstance(current_parameter, Tensor):
                    parameter_idx = system.get_parameter_names().index(parameter)
                else:
                    parameter_idx = parameter
                current_parameter[parameter_idx] += self.change_value[
                                                        self.change_parameters.index(parameter)] * np.random.choice(
                    [-1, 1], 1).item()
            current_parameter = clip_parameters(current_parameter, self.init_sample_ranges,
                                                system.get_parameter_names(),
                                                parameter_warparound=self.parameter_warparound)
            system.set_parameter(current_parameter[0])

        self.step += 1
