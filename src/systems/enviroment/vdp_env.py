from typing import Tuple, Callable, List, Union, Sequence

import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from torch import Tensor
from torch.nn.functional import pad
from torch.types import Device

from .env import Env, StateCost, TerminalCost
from .init_state_env import InitStateEnv
from ..planner.planner import get_circle_trajectory
from src.systems.systems.van_der_pol import VanDerPol
from ..runner.interactors import Interactor
from ...data.dataset.base import BasePckHdf5Loader


def circle_distance_cost(trajectories: Float[Tensor, "batch time 2"]) \
        -> Float[Tensor, "batch time"]:
    """ Calculates the distance cost between the current state and a circle with radius 2

    Args:
        trajectories: Current state; shape(batch_size, time, 2)
    Returns:
        distance_cost: Distance cost; shape(batch_size, time)
    """
    circle_distance = torch.abs(torch.sqrt(trajectories[:, :, 0] ** 2 + trajectories[:, :, 1] ** 2) - 2)
    return circle_distance


def circle_speed_cost(trajectories: Float[Tensor, "batch time 2"]) \
        -> Float[Tensor, "batch time"]:
    """ Calculates the speed cost with a goal speed that a circle is traversed in 2 seconds
    Args:
        trajectories: Current state; shape(batch_size, time, 2)

    Returns:
        speed_cost: Speed cost; shape(batch_size, time)
    """
    """
            Calculates the speed cost with a goal speed that a circle is traversed in 2 seconds
            Args:
                trajectories: Current state; shape(batch_size, time, 2)

            Returns:
                speed_cost: Speed cost; shape(batch_size, time)
            """
    # circle_size = torch.pi * 2 * 2
    # circle_time = 2
    # target_step_length = circle_size / (circle_time / self.dt)

    # speed = torch.abs(torch.norm(trajectories[:, :-1, :] - trajectories[:, 1:, :], dim=-1) - target_step_length)

    # calculate angular postion
    theta = torch.atan2(trajectories[:, :, 1], trajectories[:, :, 0])

    speed = theta[:, :-1] - theta[:, 1:]
    speed[speed < - 6] += 2 * torch.pi
    cost = torch.nn.functional.relu((0.001 - speed))
    cost = pad(input=cost, pad=(1, 0), mode='constant', value=0)
    return cost


class VdpEnv(InitStateEnv):
    """  Vdp Environment

    Ref :
        https://ocw.mit.edu/courses/
        electrical-engineering-and-computer-science/
        6-832-underactuated-robotics-spring-2009/readings/
        MIT6_832s09_read_ch03.pdf
    """

    def __init__(self,
                 pre_generated_data: BasePckHdf5Loader,
                 interactor: Interactor,
                 max_steps: int = None,
                 device: str = "cpu",
                 process_noise: float = 0.01,
                 observation_noise: float = 0.001):
        """
        """
        self.state_size = 2
        self.control_size = 1
        super(VdpEnv, self).__init__(self.state_size,
                                     pre_generated_data,
                                     interactor,
                                     max_steps,
                                     device,
                                     process_noise,
                                     observation_noise
                                     )

        self.system = VanDerPol(self.dt)

        self.input_lower_bound = torch.tensor([-4.])
        self.input_upper_bound = torch.tensor([4.])

    def get_goal(self):
        """
        Generates the goal trajectory
        For this example its a circle with radius 2
        Returns:
            goal: Goal trajectory; shape(time, 2)
        """
        return get_circle_trajectory(2, 400, 1, device=self.device).repeat(len(self.init_idx), 1, 1)

    @staticmethod
    def plot_func(to_plot, i=None, history_s=None, batch_size: int = None, labels: Union[Sequence[str], None] = None):
        """ plot cartpole object function

        Args:
            to_plot: plotted objects
            i: frame count
            history_x: history of state, shape(batch, iters, state)
            batch_size: batch size
        Returns:
            None or imgs : imgs order is ["cart_img", "pole_img"]
        """
        if isinstance(to_plot, Axes):
            assert batch_size < 7, "batch size must be less than 7"
            color_list = ["r", "g", "b", "c", "m", "y", "w"]
            imgs = {}  # create new imgs

            for batch_nr in range(batch_size):
                if labels is not None:
                    imgs[f"{batch_nr}center"] = to_plot.plot([], [], marker="o", color=color_list[batch_nr],
                                                             markersize=10, label=labels[batch_nr])[0]
                else:
                    imgs[f"{batch_nr}center"] = to_plot.plot([], [], marker="o", color=color_list[batch_nr],
                                                             markersize=10)[0]
                for j in range(20):
                    imgs[f"{batch_nr}center{j}"] = to_plot.plot([], [], marker="o", color=color_list[batch_nr],
                                                                markersize=1)[0]
            # circle
            circle = plt.Circle((0, 0), 2, color='k', fill=False)
            to_plot.add_patch(circle)

            # set axis
            to_plot.set_xlim([-4., 4])
            to_plot.set_ylim([-4, 4])

            return imgs
        for batch_nr in range(history_s.size(0)):
            to_plot[f"{batch_nr}center"].set_data(history_s[batch_nr, i, 0], history_s[batch_nr, i, 1])
            for j in range(20):
                b = max(i - j, 0)
                to_plot[f"{batch_nr}center{j}"].set_data(history_s[batch_nr, b, 0], history_s[batch_nr, b, 1])

    def _costs(self,
              history_s: List[Float[Tensor, "batch 2"]],
              history_u: List[Float[Tensor, "batch 1"]],
              ) -> Float[Tensor, "batch costs"]:
        """
        Calculates the cost of the last step
        can be calculated if two steps were made. Otherwise, it is 0

        The costs are the distance to the circle and the speed

        Args:
            history_s: History of states; shape(time, batch, state_size)

        Returns:
            costs: Costs; shape(batch)
        """
        if len(history_s) < 2:
            distance = torch.zeros(history_s[0].shape[0], device=history_s[0].device)
            speed = torch.zeros(history_s[0].shape[0], device=history_s[0].device)
            total_cost = distance + speed
            return torch.stack([total_cost, distance, speed], dim=1)
        else:
            distance_cost = circle_distance_cost(history_s[-1][:, None])
            speed_cost = circle_speed_cost(torch.stack([history_s[-2], history_s[-1]], dim=1))[:, 1, None]
            total_cost = distance_cost + speed_cost
            return torch.concatenate([total_cost, distance_cost, speed_cost], dim=1)


class CircleDistanceCost(StateCost):
    def cost(self, trajectories: Float[Tensor, "batch time 2"], goal: Float[Tensor, "batch time 2"]) \
            -> Float[Tensor, "batch time"]:
        """ Calculates the distance cost between the current state and a circle with radius 2

        Args:
            trajectories: Current state; shape(batch_size, time, 2)
        Returns:
            distance_cost: Distance cost; shape(batch_size, time)
        """

        return circle_distance_cost(trajectories)


class CircleSpeedCost(StateCost):
    def cost(self, trajectories: Float[Tensor, "batch time 2"], goal: Float[Tensor, "batch time 2"]) \
            -> Float[Tensor, "batch time"]:
        """
        Calculates the speed cost with a goal speed that a circle is traversed in 2 seconds
        Args:
            trajectories: Current state; shape(batch_size, time, 2)

        Returns:
            speed_cost: Speed cost; shape(batch_size, time)
        """
        return circle_speed_cost(trajectories)


class CircleDistanceTerminalCost(TerminalCost):

    def __init__(self, weight: float = 100):
        super().__init__()
        self.weight = weight

    def cost(self, s: Float[Tensor, "batch 2"],
             goal: Float[Tensor, "batch 2"]) \
            -> Float[Tensor, "batch"]:
        """
        Calculates the terminal state costs for some trajectories
        Args:
            s: terminal state; shape(batch_size, 2)
            goal: goal state; shape(batch_size, 2)

        Returns:
            terminal_costs: Terminal costs; shape(batch_size, sample)
        """
        return self.weight * torch.abs(torch.sqrt(s[:, 0] ** 2 + s[:, 1] ** 2) - 2)
