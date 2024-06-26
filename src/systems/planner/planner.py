import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from torch.types import Device


class Planner():
    """
    """

    def __init__(self, device: str = "cpu"):
        """
        """
        self.device = torch.device(device)

    def to(self, device: Device):
        self.device = device

    def plan(self,
             curr_s: Float[Tensor, "batch state"],
             g_traj: Float[Tensor, "batch time state"]):
        """
        Planes the next goal state in the horizon based on the global goal trajectory
        Args:
            curr_s: Current state; shape(state_size)
            g_traj: Global goal trajectory; shape(time, state_size)

        Returns:

        """
        raise NotImplementedError("Implement plan func")


class NotPlanner(Planner):
    def __init__(self, length: int, device: str = "cpu"):
        """
        Length is the length of step that should be planned
        The length of the trajectory is length
        The state were the agent should be at the moment is not included
        """
        super().__init__(device)
        self.length = length

    def plan(self,
             curr_s: Float[Tensor, "batch state"],
             g_traj: Float[Tensor, "batch time state"]) -> Float[Tensor, "batch length state"]:
        """
        Returns a fixed sized none meaningful trajectory
        """
        return torch.zeros((curr_s.size(0), self.length, curr_s.size(1)), device=curr_s.device)

def get_circle_trajectory(radius, steps, circles=1, device="cpu"):
    rad = torch.arange(0, torch.pi*2*circles, torch.pi*2*circles/steps, device=device)
    x = radius*torch.cos(rad)[:, None]
    y = radius*torch.sin(rad)[:, None]
    return torch.concatenate([y, x], dim=1)