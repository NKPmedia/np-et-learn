from typing import List, Sequence, Union

import numpy as np
import torch
from jaxtyping import Float
from matplotlib.axes import Axes
from torch import Tensor

from src.data.dataset.base import BasePckHdf5Loader
from src.systems.enviroment.env import TerminalCost, StateCost, InputCost
from src.systems.enviroment.init_state_env import InitStateEnv
from src.systems.runner.interactors import Interactor
from src.systems.systems.inverted_pendulum import InvertedPendulum
from src.utils.visualization.plot_helpers import square


class InvertedPendulumEnv(InitStateEnv):
    """  InvertedPendulum Environment

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

        self.state_size = 4
        self.control_size = 1
        super(InvertedPendulumEnv, self).__init__(self.state_size,
                                                  pre_generated_data,
                                                  interactor,
                                                  max_steps,
                                                  device,
                                                  process_noise,
                                                  observation_noise)

        self.system = InvertedPendulum(self.dt)
        self.input_lower_bound = torch.tensor([-5.])
        self.input_upper_bound = torch.tensor([5.])

        self.goal_state = torch.tensor([[0., 0., 0., 0]], device=self.device)

    def to(self, device: torch.device):
        super().to(device)
        self.goal_state = self.goal_state.to(device)

    def get_goal(self):
        """
        Generates the goal trajectory
        For this example it's the still upright position of the pendulum
        Returns:
            goal: Goal trajectory; shape(time, 4)
        """
        return torch.tile(self.goal_state, (100, 1))

    def _costs(self,
              history_s: List[Float[Tensor, "batch 4"]],
              history_u: List[Float[Tensor, "batch 1"]]) -> Float[Tensor, "batch costs"]:
        """
        Calculates the costs for the last state and input for an inverted pendulum

        Args:
            history_s: History of states; shape(time, batch, state_size)
            history_u: History of inputs; shape(time, batch, input_size)

        Returns:
            costs: Costs; shape(batch)
        """
        u = history_u[-1]
        state = history_s[-1]

        cost_u = 0.1 * torch.sum(u ** 2, dim=-1)
        cost_state = 6. * state[:, 0] ** 2 \
                     + 12. * (1 - torch.cos(state[:, 2])) ** 2 \
                     + 0.1 * state[:, 1] ** 2 \
                     + 0.1 * state[:, 3] ** 2
        cost_u = cost_u.unsqueeze(-1)
        cost_state = cost_state.unsqueeze(-1)

        total_cost = cost_u + cost_state
        return torch.concatenate([total_cost, cost_u, cost_state], dim=1)

    @staticmethod
    def plot_func(to_plot, i=None, history_s=None, batch_size: int = None, labels: Union[Sequence[str], None] = None):
        """ plot cartpole object function

        Args:
            to_plot: plotted objects
            i: frame count
            history_x: history of state, shape(iters, state)
            history_g_x: history of goal state,
                                         shape(iters, state)
        Returns:
            None or imgs : imgs order is ["cart_img", "pole_img"]
        """
        if isinstance(to_plot, Axes):
            assert batch_size < 7, "batch size must be less than 7"
            color_list = ["r", "g", "b", "c", "m", "y", "w"]
            imgs = {}  # create new imgs

            for batch_nr in range(batch_size):
                if labels is not None:
                    imgs[f"{batch_nr}cart"] = to_plot.plot([], [], color=color_list[batch_nr], label=labels[batch_nr])[0]
                else:
                    imgs[f"{batch_nr}cart"] = to_plot.plot([], [], color=color_list[batch_nr])[0]
                imgs[f"{batch_nr}pole"] = to_plot.plot([], [], color=color_list[batch_nr], linewidth=5)[0]
                imgs[f"{batch_nr}center"] = to_plot.plot([], [], marker="o", color=color_list[batch_nr],
                                                         markersize=10)[0]
            # centerline
            to_plot.plot(np.linspace(-1., 1., num=50), np.zeros(50),
                         c="k", linestyle="dashed")

            # set axis
            to_plot.set_xlim([-1., 1.])
            to_plot.set_ylim([-0.55, 1.5])

            return imgs

            # set imgs
        for batch_nr in range(history_s.size(0)):
            cart_x, cart_y, pole_x, pole_y = \
                InvertedPendulumEnv._plot_cartpole(history_s[batch_nr, i])

            to_plot[f"{batch_nr}cart"].set_data(cart_x, cart_y)
            to_plot[f"{batch_nr}pole"].set_data(pole_x, pole_y)
            to_plot[f"{batch_nr}center"].set_data(history_s[batch_nr, i, 0], 0.)

    @staticmethod
    def _plot_cartpole(curr_x):
        """ plot cartpole fucntions

        Args:
            curr_x (numpy.ndarray): current catpole state
        Returns:
            cart_x (numpy.ndarray): x data of cart
            cart_y (numpy.ndarray): y data of cart
            pole_x (numpy.ndarray): x data of pole
            pole_y (numpy.ndarray): y data of pole
        """
        # cart
        cart_x, cart_y = square(curr_x[0], 0., (0.15, 0.1), 0.)

        L = 0.2
        # pole
        pole_x = np.array([curr_x[0], curr_x[0] + L
                           * np.sin(curr_x[2])])
        pole_y = np.array([0., L
                           * np.cos(curr_x[2])])

        return cart_x, cart_y, pole_x, pole_y


class InvPInputCost(InputCost):
    def cost(self, input: Float[Tensor, "batch time 1"]):
        """
        Calculates the input costs for some trajectories
        Args:
            input: Input; shape(batch_size, time, 1)
        Returns:
            input_costs: Trajectory costs; shape(batch_size, time)
        """
        return torch.sum(input ** 2, dim=(2)) * 0.01


class InvPStateCost(StateCost):
    def cost(self,
             trajectories: Float[Tensor, "batch time 4"],
             g: Float[Tensor, "batch time 4"]):
        """
        Calculates the state costs for some trajectories
        Args:
            trajectories: State trajectories; shape(batch_size, time, 4)
            g: Goal trajectories; shape(batch_size, time, 4)
        Returns:
            trajectory_costs: Trajectory costs; shape(batch_size, time)
        """
        cost = 6. * (trajectories[:, :, 0] ** 2) \
               + 12. * ((1 - torch.cos(trajectories[:, :, 2])) ** 2) \
               + 0.1 * (trajectories[:, :, 1] ** 2) \
               + 0.1 * (trajectories[:, :, 3] ** 2)
        return cost


class InvPTerminalCost(TerminalCost):
    def cost(self,
             s: Float[Tensor, "batch 4"],
             goal: Float[Tensor, "batch 4"]):
        """
        Calculates the terminal state costs for some trajectories
        Args:
            s: terminal state; shape(batch_size, 4)
            goal: goal state; shape(batch_size, 4)

        Returns:
            terminal_costs: Terminal costs; shape(batch_size)
        """
        return 6. * (s[:, 0] ** 2) \
            + 12. * ((1 - torch.cos(s[:, 2])) ** 2) \
            + 0.1 * (s[:, 1] ** 2) \
            + 0.1 * (s[:, 3] ** 2)
