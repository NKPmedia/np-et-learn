import math
from math import sqrt, log
from typing import List, Sequence

import torch
from jaxtyping import Float, Bool
from torch import Tensor


class ResetTrigger:

    def reset(self, batch_size: int):
        pass

    def to(self, device):
        self.device = device

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        raise NotImplementedError("Implement test_trigger function")


class PlotableResetTrigger(ResetTrigger):

    def get_bound(self,
                  pred_states: Float[Tensor, "batch time state"],
                  pred_std: Float[Tensor, "batch time state"],
                  observed_states: Float[Tensor, "batch time state"],
                  step: int
                  ):
        raise NotImplementedError("Implement test_trigger function")

    def get_observed_values(self,
                            pred_states: Float[Tensor, "batch time state"],
                            pred_std: Float[Tensor, "batch time state"],
                            observed_states: Float[Tensor, "batch time state"],
                            step: int
                            ):
        raise NotImplementedError("Implement test_trigger function")

class ConstantTrigger(ResetTrigger):
    """
    Triggers a reset at a given reset_steps
    """

    def __init__(self, reset_steps: List[int]):
        self.reset_steps = reset_steps

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        """
        s_init -> s0 -> s1 -> s2 -> s3 -c> s4 -> s5
        We trigger after the computation of s4.
        The length of the history is one larger than the step becuase we already added the comuted step to the history.
        Before we test for triggers.
        Therefore, we test if step is in reset_steps.
        Args:
            pred_states:
            pred_std:
            observed_states:
            step:

        Returns:

        """
        return torch.tensor([step in self.reset_steps for _ in range(pred_states.size(0))])


class GaussianBoundTrigger(ResetTrigger):
    """
    Uses the predicted gaussian distribution to determine if a reset should be triggered
    Tests if the observed state is outside a defined confidence interval
    """

    def __init__(self, prop: float, cool_down: int = 0):
        self.prop = prop
        self.cool_down = cool_down
        self.last_trigger_step = None

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        assert self.last_trigger_step is not None, "ResetTrigger.reset() must be called before test_trigger()"

        bound_size = sqrt(2) * pred_std[:, -1] * torch.erfinv(torch.Tensor([self.prop])).item()
        difference = torch.abs(pred_states[:, -1] - observed_states[:, -1])

        trigger = difference > bound_size
        trigger = trigger.sum(dim=-1).to(torch.bool).flatten()
        trigger = trigger & (step - self.last_trigger_step > self.cool_down)

        self.last_trigger_step[trigger] = step

        return trigger


class GaussianChernoffTrigger(ResetTrigger):
    """
    Uses the chernoff bound to determine if a reset should be triggered.
    Assumes a gaussian distribution of the predicted error.
    """

    def __init__(self, prop: float,
                 cool_down: int = 0,
                 min_triggers: int = 1,
                 reset_after: int = 1,
                 min_channel_triggers: int = 1,
                 ignore_trigger_steps: Sequence[int] = [],
                 save_trigger_channel: bool = False,
                 time_bound_adaption: bool = False):
        self.prop = prop
        self.cool_down = cool_down
        self.min_triggers = min_triggers
        self.reset_after = reset_after
        self.last_trigger_step = None
        self.ignore_trigger_steps = ignore_trigger_steps
        self.trigger_count = None
        self.min_channel_triggers = min_channel_triggers
        self.save_trigger_channel = save_trigger_channel
        self.time_bound_adaption = time_bound_adaption

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.last_reported_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_count = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_channel = []

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        assert self.last_trigger_step is not None, "ResetTrigger.reset() must be called before test_trigger()"

        if step in self.ignore_trigger_steps:
            return torch.zeros(pred_states.size(0), dtype=torch.bool, device=self.device)

        time_since_reported_trigger = step - self.last_reported_trigger_step
        if self.time_bound_adaption:
            pi_t = torch.pi ** 2 * time_since_reported_trigger ** 2 / 6
            pi_t = pi_t[:, None].tile((1, pred_std.size(2)))
        else:
            pi_t = torch.ones((time_since_reported_trigger.size(0), pred_std.size(2)), device=self.device)
        prop_out = 1 - self.prop
        bound_size = torch.sqrt(
            -2 * pred_std[:, -1] ** 2 * torch.log(1 / pi_t * torch.tensor([prop_out / 2], device=self.device)))

        difference = torch.abs(pred_states[:, -1] - observed_states[:, -1])

        trigger = difference > bound_size
        if self.save_trigger_channel:
            self.trigger_channel.append(trigger.clone())
        trigger = trigger.sum(dim=-1).flatten()
        trigger = trigger >= self.min_channel_triggers
        trigger = trigger.to(torch.bool)

        trigger = trigger & (step - self.last_trigger_step > self.cool_down)

        self.last_trigger_step[trigger] = step

        self.trigger_count[trigger] += 1

        reported_trigger = self.trigger_count >= self.min_triggers
        # reset reported triggers to 0
        self.trigger_count[reported_trigger] = 0
        # reset trigger count for indexes which last trigger is longer ago than reset_after
        self.trigger_count[step - self.last_trigger_step > self.reset_after] = 0

        self.last_reported_trigger_step[reported_trigger] = step

        return reported_trigger


class SubGaussianNormTrigger(PlotableResetTrigger):
    """
    Uses the chernoff bound to determine if a reset should be triggered.
    Assumes a gaussian distribution of the predicted error.
    """

    def __init__(self, prop: float,
                 min_triggers: int = 1,
                 reset_after: int = 1,
                 min_channel_triggers: int = 1,
                 ignore_trigger_steps: Sequence[int] = [],
                 save_trigger_channel: bool = False,
                 bound_version: str = "original",
                 time_bound_adaption: bool = False):
        self.prop = prop
        self.min_triggers = min_triggers
        self.reset_after = reset_after
        self.last_trigger_step = None
        self.ignore_trigger_steps = ignore_trigger_steps
        self.trigger_count = None
        self.min_channel_triggers = min_channel_triggers
        self.save_trigger_channel = save_trigger_channel
        self.bound_version = bound_version
        self.time_bound_adaption = time_bound_adaption

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.last_reported_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_count = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_channel = []

    def get_bound(self,
                  pred_states: Float[Tensor, "batch time state"],
                  pred_std: Float[Tensor, "batch time state"],
                  observed_states: Float[Tensor, "batch time state"],
                  step: int
                  ):

        time_since_reported_trigger = step - self.last_reported_trigger_step
        if self.time_bound_adaption:
            pi_t = torch.pi ** 2 * time_since_reported_trigger ** 2 / 6
        else:
            pi_t = torch.ones((time_since_reported_trigger.size(0)), device=self.device)

        maximal_sigma = torch.max(pred_std[:, -1], dim=-1)[0]
        dimensions = pred_states.size(-1)
        prop_out = 1 - self.prop
        if self.bound_version == "original":
            bound = 4 * maximal_sigma * sqrt(dimensions) \
                    + 2 * maximal_sigma * torch.sqrt(torch.log(1 / torch.tensor([prop_out], device=self.device)))
        elif self.bound_version == "tight":
            bound = torch.sqrt(8 * maximal_sigma ** 2 * torch.log(5**dimensions * pi_t / torch.tensor([prop_out], device=self.device)))
        return bound[:, None]

    def get_observed_values(self,
                            pred_states: Float[Tensor, "batch time state"],
                            pred_std: Float[Tensor, "batch time state"],
                            observed_states: Float[Tensor, "batch time state"],
                            step: int
                            ):
        difference = torch.abs(pred_states[:, -1] - observed_states[:, -1])
        diff_norm = torch.norm(difference, dim=-1)
        return diff_norm[:, None]

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        assert self.last_trigger_step is not None, "ResetTrigger.reset() must be called before test_trigger()"

        if step in self.ignore_trigger_steps:
            return torch.zeros(pred_states.size(0), dtype=torch.bool, device=self.device)

        bound = self.get_bound(pred_states, pred_std, observed_states, step)
        observed_values = self.get_observed_values(pred_states, pred_std, observed_states, step)

        trigger = observed_values > bound
        if self.save_trigger_channel:
            self.trigger_channel.append(trigger.clone())
        trigger = trigger.sum(dim=-1).flatten()
        trigger = trigger >= self.min_channel_triggers
        trigger = trigger.to(torch.bool)

        self.last_trigger_step[trigger] = step

        self.trigger_count[trigger] += 1

        reported_trigger = self.trigger_count >= self.min_triggers
        # reset reported triggers to 0
        self.trigger_count[reported_trigger] = 0
        # reset trigger count for indexes which last trigger is longer ago than reset_after
        self.trigger_count[step - self.last_trigger_step > self.reset_after] = 0

        self.last_reported_trigger_step[reported_trigger] = step

        return reported_trigger


class GaussianNormTrigger(SubGaussianNormTrigger):
    """
    Uses the chernoff bound to determine if a reset should be triggered.
    Assumes a gaussian distribution of the predicted error.
    """

    def __init__(self, prop: float,
                 min_triggers: int = 1,
                 reset_after: int = 1,
                 min_channel_triggers: int = 1,
                 ignore_trigger_steps: Sequence[int] = [],
                 save_trigger_channel: bool = False,
                 time_bound_adaption: bool = False,
                 version: str = "original"):
        self.prop = prop
        self.min_triggers = min_triggers
        self.reset_after = reset_after
        self.last_trigger_step = None
        self.ignore_trigger_steps = ignore_trigger_steps
        self.trigger_count = None
        self.min_channel_triggers = min_channel_triggers
        self.save_trigger_channel = save_trigger_channel
        self.time_bound_adaption = time_bound_adaption
        self.version = version

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.last_reported_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_count = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_channel = []

    def get_bound(self,
                  pred_states: Float[Tensor, "batch time state"],
                  pred_std: Float[Tensor, "batch time state"],
                  observed_states: Float[Tensor, "batch time state"],
                  step: int
                  ):

        time_since_reported_trigger = step - self.last_reported_trigger_step
        if self.time_bound_adaption:
            pi_t = torch.pi ** 2 * time_since_reported_trigger ** 2 / 6
        else:
            pi_t = torch.ones((time_since_reported_trigger.size(0)), device=self.device)

        sigma_sum = torch.sum(pred_std[:, -1]**2, dim=-1)

        prop_out = 1 - self.prop

        if self.version == "original":
            bound = torch.sqrt(8 * sigma_sum * torch.log(sqrt(math.e) * pi_t/ torch.tensor([prop_out], device=self.device)))
        elif self.version == "tight":
            bound = torch.sqrt(2 * sigma_sum * torch.log(2 * pi_t/ torch.tensor([prop_out], device=self.device)))
        else:
            raise NotImplementedError("Version {} not implemented".format(self.version))
        return bound[:, None]

class GaussianSrinivasTrigger(PlotableResetTrigger):
    """
    Uses the chernoff bound to determine if a reset should be triggered.
    Assumes a gaussian distribution of the predicted error.
    """

    def __init__(self, prop: float,
                 min_triggers: int = 1,
                 reset_after: int = 1,
                 min_channel_triggers: int = 1,
                 ignore_trigger_steps: Sequence[int] = [],
                 save_trigger_channel: bool = False,
                 time_bound_adaption: bool = False):
        self.prop = prop
        self.min_triggers = min_triggers
        self.reset_after = reset_after
        self.last_trigger_step = None
        self.ignore_trigger_steps = ignore_trigger_steps
        self.trigger_count = None
        self.min_channel_triggers = min_channel_triggers
        self.save_trigger_channel = save_trigger_channel
        self.time_bound_adaption = time_bound_adaption

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.last_reported_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_count = torch.zeros(batch_size, dtype=torch.int, device=self.device)
        self.trigger_channel = []

    def get_bound(self,
                  pred_states: Float[Tensor, "batch time state"],
                  pred_std: Float[Tensor, "batch time state"],
                  observed_states: Float[Tensor, "batch time state"],
                  step: int
                  ):
        time_since_reported_trigger = step - self.last_reported_trigger_step
        if self.time_bound_adaption:
            pi_t = torch.pi ** 2 * time_since_reported_trigger ** 2 / 6
            pi_t = pi_t[:, None].tile((1, pred_std.size(2)))
        else:
            pi_t = torch.ones((time_since_reported_trigger.size(0), pred_std.size(2)), device=self.device)

        prop_out = 1 - self.prop
        bound = torch.sqrt(2 * pred_std[:, -1] ** 2 * torch.log(pi_t / torch.tensor([prop_out], device=self.device)))

        return bound

    def get_observed_values(self,
                            pred_states: Float[Tensor, "batch time state"],
                            pred_std: Float[Tensor, "batch time state"],
                            observed_states: Float[Tensor, "batch time state"],
                            step: int
                            ):
        difference = torch.abs(pred_states[:, -1] - observed_states[:, -1])
        return difference

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        assert self.last_trigger_step is not None, "ResetTrigger.reset() must be called before test_trigger()"

        if step in self.ignore_trigger_steps:
            return torch.zeros(pred_states.size(0), dtype=torch.bool, device=self.device)

        bound = self.get_bound(pred_states, pred_std, observed_states, step)
        observed_values = self.get_observed_values(pred_states, pred_std, observed_states, step)

        trigger = observed_values > bound
        if self.save_trigger_channel:
            self.trigger_channel.append(trigger.clone())
        trigger = trigger.sum(dim=-1).flatten()
        trigger = trigger >= self.min_channel_triggers
        trigger = trigger.to(torch.bool)

        self.last_trigger_step[trigger] = step

        self.trigger_count[trigger] += 1

        reported_trigger = self.trigger_count >= self.min_triggers
        # reset reported triggers to 0
        self.trigger_count[reported_trigger] = 0
        # reset trigger count for indexes which last trigger is longer ago than reset_after
        self.trigger_count[step - self.last_trigger_step > self.reset_after] = 0

        self.last_reported_trigger_step[reported_trigger] = step

        return reported_trigger


class HoeffdingTrigger(PlotableResetTrigger):
    """
    Uses the hoeffding bound to determine if a reset should be triggered.

    """

    def __init__(self, prop: float, cool_down: int = 0, assumed_mean: float = 0, window_size: int = 5):
        self.prop = prop
        self.cool_down = cool_down
        self.assumed_mean = assumed_mean
        self.last_trigger_step = None
        self.window_size = window_size

    def to(self, device):
        self.device = device
        if self.last_trigger_step is not None:
            self.last_trigger_step = self.last_trigger_step.to(device)

    def reset(self, batch_size: int):
        self.last_trigger_step = torch.zeros(batch_size, dtype=torch.int, device=self.device)

    def get_bound(self,
                  pred_states: Float[Tensor, "batch time state"],
                  pred_std: Float[Tensor, "batch time state"],
                  observed_states: Float[Tensor, "batch time state"],
                  step: int
                  ):
        prop_out = 1 - self.prop

        min_idx = max(0, step - self.window_size)
        std = pred_std[:, min_idx:]
        std_squared = std ** 2
        std_squared_sum = torch.sum(std_squared, dim=1)

        length = std_squared.size(1)

        bound = torch.sqrt(-(2 * log(prop_out / 2) * std_squared_sum) / (length ** 2))

        return bound

    def get_observed_values(self,
                            pred_states: Float[Tensor, "batch time state"],
                            pred_std: Float[Tensor, "batch time state"],
                            observed_states: Float[Tensor, "batch time state"],
                            step: int
                            ):
        min_idx = max(0, step - self.window_size)
        std = pred_std[:, min_idx:]
        std_squared = std ** 2
        abs_error_sum = torch.abs(torch.sum(pred_states[:, min_idx:] - observed_states[:, min_idx:], dim=1))

        length = std_squared.size(1)

        trigger_val = abs_error_sum / length
        return trigger_val

    def test_trigger(self,
                     pred_states: Float[Tensor, "batch time state"],
                     pred_std: Float[Tensor, "batch time state"],
                     observed_states: Float[Tensor, "batch time state"],
                     step: int) -> \
            Bool[Tensor, "batch"]:
        assert self.last_trigger_step is not None, "ResetTrigger.reset() must be called before test_trigger()"

        bound = self.get_bound(pred_states, pred_std, observed_states, step)
        trigger_val = self.get_observed_values(pred_states, pred_std, observed_states, step)

        trigger = trigger_val > bound
        trigger = trigger.sum(dim=-1).to(torch.bool).flatten()
        trigger = trigger & (step - self.last_trigger_step > self.cool_down)

        self.last_trigger_step[trigger] = step

        return trigger
