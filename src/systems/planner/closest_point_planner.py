import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from .planner import Planner


class ClosestPointPlanner(Planner):
    """ This planner generates a local goal trajectory by finding the closest point on the global goal trajectory
    """

    def __init__(self, pred_len, n_ahead, path_is_circle: bool=True):
        """
        """
        super(ClosestPointPlanner, self).__init__()
        self.pred_len = pred_len
        self.n_ahead = n_ahead
        self.path_is_circle = path_is_circle
        assert path_is_circle, "Only circle path is supported at the moment"

    def plan(self,
             curr_s: Float[Tensor, "batch state"],
             g_traj: Float[Tensor, "batch time state"]) ->\
            Float[Tensor, "batch local_time state"]:
        """
        Generate a local goal trajectory by finding the closest point on the global goal trajectory the resulting
        trajectory has length pred_len+1
        Args:
            curr_s: Current state; shape(state_size)
            g_traj: Global goal trajectory; shape(time, state_size)

        Returns:
            goal trajectory; shape(pred_len+1, state_size)
        """
        #TODO fix with mutlibatch!!!!
        if g_traj.ndim == 2:
            g_traj = torch.tile(g_traj[None, :, :], (curr_s.shape[0], 1, 1))
        min_idx = torch.stack([torch.argmin(torch.norm(curr_s[i, None] - g_traj[i],
                                           dim=1)) for i in range(curr_s.shape[0])])
        if self.path_is_circle:
            g_traj_extended = g_traj.repeat(1, 2, 1)
        else:
            g_traj_extended = g_traj

        start = (min_idx+self.n_ahead)
        end = min_idx+self.n_ahead+self.pred_len+1

        plans = [g_traj_extended[i, start[i]: end[i]] for i in range(curr_s.shape[0])]

        return torch.stack(plans)