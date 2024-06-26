from types import NoneType
from typing import Callable, Tuple, Union, Sequence, List

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.types import Device

from src.data.dataset.base import BasePckHdf5Loader
from src.systems.enviroment.env import Env
from src.systems.runner.interactors import Interactor


class InitStateEnv(Env):
    """ Environments class
    """

    def __init__(self,
                 state_size: int,
                 pre_generated_data: BasePckHdf5Loader,
                 interactor: Interactor,
                 max_steps: int = None,
                 device: str = "cpu",
                 process_noise: float = 0.01,
                 observation_noise: float = 0.001,
                 ):
        """
        """
        assert pre_generated_data is not None, "pre_generated_data must be set. other versions are not supported anymore"

        if max_steps is None:
            max_steps = pre_generated_data["parameter"].size(1)

        super().__init__(state_size, max_steps, device, process_noise, observation_noise)

        self.system = None
        self.interactor = interactor
        self.input_lower_bound = torch.tensor([-999999.], device=self.device)
        self.input_upper_bound = torch.tensor([999999.], device=self.device)

        self.init_idx = [0]
        self.pre_generated_data = pre_generated_data
        self.pre_generated_data.to(self.device)
        self.reset_noise()

    def to(self, device: torch.device):
        super().to(device)
        self.input_lower_bound = self.input_lower_bound.to(device)
        self.input_upper_bound = self.input_upper_bound.to(device)
        self.precomputed_pNoise = self.precomputed_pNoise.to(device)
        self.precomputed_oNoise = self.precomputed_oNoise.to(device)
        self.pre_generated_data.to(device)
        self.interactor.to(device)

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
        self.step_count = 0

        self.select_init(init_idx)
        self.curr_s = self.pre_generated_data["x"][self.init_idx]

        # clear memory
        self.history_s = []  # The real state of the system
        self.history_y = []  # The observed state of the system
        self.history_u = []  # The control input

        self.interactor.reset(len(self.init_idx), channel_size=self.state_size)

        return self.curr_s

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

        assert total_step == self.step_count, "total_step must be equal to step_count"

        # clip action
        if self.input_lower_bound is not None:
            u = torch.clip(u, self.input_lower_bound, self.input_upper_bound)

        self.system.set_parameter(self.pre_generated_data["parameter"][self.init_idx, total_step])

        # step
        # x
        next_s = self.system.next_state_rk4(self.curr_s, u)

        # Add human interaction or external disturbance
        next_s += self.interactor.get_interactions(total_step)

        # Add pregenerated process noise and observation noise
        next_s += self.precomputed_pNoise[self.init_idx, self.step_count] * torch.abs(next_s - self.curr_s)
        next_y = torch.clone(next_s) + self.precomputed_oNoise[self.init_idx, self.step_count]

        # save history
        self.history_s.append(next_s)
        self.history_y.append(next_y)
        self.history_u.append(u)

        costs = self._costs(self.history_s, self.history_u)

        # update
        self.curr_s = next_s
        # update costs
        self.step_count += 1

        return next_y, \
            costs, \
            self.step_count >= (self.max_step - 1)

    def overwrite_state(self, state: Float[Tensor, "batch state"]):
        if state.ndim == 1:
            state = state[None]
            state = state.repeat(self.curr_s.size(0), 1)
        self.curr_s = state

    def reset_noise(self):
        r_state = torch.random.get_rng_state().clone()
        torch.manual_seed(42)
        self.precomputed_pNoise = torch.normal(0.,
                                               self.process_noise,
                                               size=(
                                                   self.pre_generated_data["x"].size(0), self.max_step,
                                                   self.state_size),
                                               device=self.device)
        self.precomputed_oNoise = torch.normal(0.,
                                               self.observation_noise,
                                               size=(
                                                   self.pre_generated_data["x"].size(0), self.max_step,
                                                   self.state_size),
                                               device=self.device)
        torch.random.set_rng_state(r_state)

    def _costs(self,
              history_s: List[Float[Tensor, "batch state"]],
              history_u: List[Float[Tensor, "batch control"]]) -> Float[Tensor, "batch costs"]:
        raise NotImplementedError("Implement cost function")