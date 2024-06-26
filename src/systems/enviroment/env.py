from types import NoneType
from typing import Callable, Tuple, Union, Sequence, List

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.types import Device

from src.data.dataset.base import BasePckHdf5Loader


class Env():
    """ Environments class
    """

    def __init__(self,
                 state_size: int,
                 max_steps: int = None,
                 device: str = "cpu",
                 process_noise: float = 0.01,
                 observation_noise: float = 0.001,
                 observation_size: int = -1,
                 ):
        """
        """
        self.state_size = state_size
        self.curr_s: Float[Tensor, "batch state"] = None
        self.goal_state = None
        self.history_s = []
        self.history_goal_s = []
        self.history_u = []
        self.step_count: int = 0
        self.device = torch.device(device)
        self.system = None

        if observation_size == -1:
            observation_size = state_size
        self.observation_size = observation_size

        self.dt = 0.02
        self.max_step = max_steps
        self.process_noise = process_noise
        self.observation_noise = observation_noise

        self.init_idx = [0]
        if max_steps is not None:
            self.max_step = max_steps

    def to(self, device: torch.device):
        self.device = device

    def select_init(self, init_idx: List[int]):
        self.init_idx = init_idx

    def reset(self, init_idx: List[int] = None):
        """ Resets the environment to a given, fixed initial state [-2, 2] or a random initial state
        If pre_generated_data is given, the initial state is gathered from the dataset based on the init_idx

        Args:
            init_s: Possible initial state; shape(batch, state_size)
            random_init: If true, the initial state is sampled randomly from a subset of the state space [-3, 3] in all dimensions
        Returns:
            curr_s: New current state; shape(batch, state_size)
        """
        raise NotImplementedError

    def step(self,
             u: Float[Tensor, "batch control"],
             total_step: int) -> \
            Tuple[Float[Tensor, "batch state"], Float[Tensor, "batch costs"], bool]:
        """
        Performs a step in the environment
        Args:
            u: Control input; shape(batch, control)
            total_step: current timestep in the episode

        Returns:
            next_y: Next observed state (includes observation noise); shape(batch, 2)
            costs: Costs; shape(batch, costs)
            done: True if the episode is done, False otherwise
        """

        raise NotImplementedError

    def get_goal(self):
        """
        Generates the goal trajectory
        Returns:
            goal: Goal trajectory; shape(time, 2)
        """
        return torch.zeros((1, self.state_size), device=self.device).repeat(len(self.init_idx), 1, 1)

    def close_video_recorder(self):
        pass


class StateCost:
    def cost(self, s: Float[Tensor, "batch time state"], g: Float[Tensor, "batch time state"]) -> \
            Float[Tensor, "batch time"]:
        raise NotImplementedError("Implement cost function")


class CombinedStateCost(StateCost):
    def __init__(self, state_cost: Sequence[StateCost], weights: Sequence[float]):
        self.state_cost = state_cost
        self.weights = weights

    def cost(self, s: Float[Tensor, "batch time state"], g: Float[Tensor, "batch time state"]) -> \
            Float[Tensor, "batch time"]:
        return sum([w * fn.cost(s, g) for w, fn in zip(self.weights, self.state_cost)])


class TerminalCost:
    def cost(self, s: Float[Tensor, "batch state"], g: Float[Tensor, "batch state"]) -> \
            Float[Tensor, "batch"]:
        raise NotImplementedError("Implement cost function")


class CombinedTerminalCost(TerminalCost):
    def __init__(self, terminal_cost_fns: Sequence[TerminalCost], weights: Sequence[float]):
        self.terminal_cost_fns = terminal_cost_fns
        self.weights = weights

    def cost(self, s: Float[Tensor, "batch state"], g: Float[Tensor, "batch state"]) -> \
            Float[Tensor, "batch"]:
        return sum([w * fn.cost(s, g) for w, fn in zip(self.weights, self.terminal_cost_fns)])


class InputCost:
    def cost(self, inp: Float[Tensor, "batch time input"]) -> \
            Float[Tensor, "batch time"]:
        raise NotImplementedError("Implement cost function")


StateCostCallable = Callable[
    [Float[Tensor, "batch time state"], Float[Tensor, "batch time state"]], Float[Tensor, "batch time"]]
InputCostCallable = Callable[[Float[Tensor, "batch time input"]], Float[Tensor, "batch time"]]
TerminalCostCallable = Callable[
    [Float[Tensor, "batch state"], Float[Tensor, "batch state"]], Float[Tensor, "batch"]]


def calc_cost(trajectories: Float[Tensor, "batch samples pred_len state"],
              input_sample: Float[Tensor, "batch samples pred_len control"],
              goal_states: Float[Tensor, "batch pred_len state"],
              state_cost_fn: Union[StateCostCallable, NoneType],
              input_cost_fn: Union[InputCostCallable, NoneType],
              terminal_state_cost_fn: Union[TerminalCostCallable, NoneType]) -> \
        Float[Tensor, "batch samples"]:
    """
    Calculates the cost of the trajectories and control inputs based on given cost functions
    Assumes that the goal is the same over the samples
    Args:
        trajectories: Trajectories to calculate the cost for; shape(batch samples pred_len state)
        input_sample: Control inputs to calculate the cost for; shape(batch samples pred_len control)
        goal_states: Goal states to achieve; shape(batch pred_len state)
        state_cost_fn: Cost function for the states
        input_cost_fn: Cost function for the control inputs
        terminal_state_cost_fn: Cost function for the terminal states
    Returns:
        cost (Tensor[batch]): Cost of the trajectories and control inputs
    """
    # state cost

    batch_size, samples, time, _ = trajectories.shape
    trajectories = trajectories.reshape(batch_size * samples, time, -1)
    goal_states = goal_states[:, None].repeat(1, samples, 1, 1).reshape(batch_size * samples, time, -1)
    input_sample = input_sample.reshape(batch_size * samples, time, -1)

    state_cost = 0.
    if state_cost_fn is not None:
        state_pred_par_cost = state_cost_fn(
            trajectories[:, :-1, :], goal_states[:, :-1, :])
        state_cost = torch.sum(state_pred_par_cost, dim=-1)
        state_cost = state_cost.reshape(batch_size, samples)

    # terminal cost
    terminal_state_cost = 0.
    if terminal_state_cost_fn is not None:
        terminal_state_cost = terminal_state_cost_fn(trajectories[:, -1, :],
                                                     goal_states[:, -1, :])
        terminal_state_cost = terminal_state_cost.reshape(batch_size, samples)

    # act cost
    act_cost = 0.
    if input_cost_fn is not None:
        act_pred_par_cost = input_cost_fn(input_sample)
        act_cost = torch.sum(act_pred_par_cost, dim=-1)
        act_cost = act_cost.reshape(batch_size, samples)

    return state_cost + terminal_state_cost + act_cost
