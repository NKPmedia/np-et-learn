from typing import Tuple, Callable, List

import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor
from torch.nn.functional import pad
from torch.types import Device

from .env import Env
from ..planner.planner import get_circle_trajectory
from src.systems.systems.van_der_pol import VanDerPol


class TrajectoryReplay(Env):
    """  Vdp Environment

    Ref :
        https://ocw.mit.edu/courses/
        electrical-engineering-and-computer-science/
        6-832-underactuated-robotics-spring-2009/readings/
        MIT6_832s09_read_ch03.pdf
    """
    def __init__(self,
                 trajectory: Float[Tensor, "batch time state"],
                 state_size: int,
                 device: str = "cpu",
                 observation_noise: float = 0.01
                 ):
        """
        """
        max_step = trajectory.shape[1]
        super(TrajectoryReplay, self).__init__(state_size,
                                               max_step,
                                               device,
                                               0,
                                               observation_noise)
        self.dt = 0.02 # Just assume that the dt is 0.02
        self.trajectory = trajectory

        self.gerenate_observation_noise()

    @property
    def control_size(self) -> int:
        return self.trajectory.shape[2] - self.state_size

    def to(self, device: torch.device):
        super().to(device)
        self.trajectory = self.trajectory.to(device)
        self.precomputed_oNoise = self.precomputed_oNoise.to(device)

    def reset(self, init_idx: List[int] = None):
        """
        Resets the environment to the initial state
        init state that is given will be ignored use init_idx instead
        Returns:
            curr_s: New current state; shape(batch, state_size)
        """
        self.step_count = 0

        self.select_init(init_idx)

        # clear memory
        self.history_s = []  # The real state of the system
        self.history_y = []  # The observed state of the system
        self.history_u = []  # The control input of the system

        self.curr_s = self.trajectory[self.init_idx, 0, :self.state_size]

        return self.curr_s

    def get_last_control(self):
        """
        Returns the last control input
        Gets the control input from the control var in the given trajectories
        Returns:
            last_control: Last control input; shape(batch, control_size)
        """
        return self.trajectory[self.init_idx, self.step_count, self.state_size:]

    def step(self,
             u: Float[Tensor, "batch control"],
             total_step: int) -> \
            Tuple[Float[Tensor, "batch state"], Float[Tensor, "batch"], bool]:
        """
        Performs a step in the environment.
        Next state is given by the given trajectory with added observation noise
        Control input is ignored
        Args:
            u: Control input; shape(batch, control_size)
            total_step: current timestep in the episode

        Returns:
            next_y: Next observed state (includes observation noise); shape(batch, state_size)
            costs: Costs; shape(batch)
            done: True if the episode is done, False otherwise
        """
        self.step_count += 1
        # step
        # x
        next_s = self.trajectory[self.init_idx, self.step_count, :self.state_size]
        next_y = torch.clone(next_s) + self.precomputed_oNoise[self.init_idx, self.step_count]

        # TODO: costs
        costs = torch.zeros(next_s.shape[0], device=self.device)

        # save history
        self.history_s.append(next_s)
        self.history_y.append(next_y)

        # update
        self.curr_s = next_s
        # update costs

        return next_y, \
            costs, \
            self.step_count >= (self.max_step - 1)

    def get_goal(self):
        return None

    def gerenate_observation_noise(self):
        """
        Generates observation noise for the current trajectory
        Returns:

        """
        r_state = torch.random.get_rng_state().clone()
        torch.manual_seed(42)
        self.precomputed_oNoise = torch.normal(0.,
                                              self.observation_noise,
                                              size=(self.trajectory.size(0), self.trajectory.size(1), self.state_size),
                                              device=self.trajectory.device)
        torch.random.set_rng_state(r_state)

    def set_data(self, data):
        self.trajectory = data