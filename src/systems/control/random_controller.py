import random
from logging import getLogger
from typing import Sequence, List

import torch
from jaxtyping import Float
from torch import Tensor

from .controller import Controller
from src.systems.systems.dynamic_system import DynamicSystem, DynamicSystemType
from ..predictor.predictor import Predictor

logger = getLogger(__name__)


class RandomController(Controller):
    """ Zero controller. Only applies zero control input
    """

    def __init__(self, predictor: Predictor,
                 system_type: DynamicSystemType,
                 control_range: Sequence[Sequence[float]],
                 one_step_predictors: List[Predictor] = None,
                 smoothing: float = 0.):
        super(RandomController, self).__init__(predictor, system_type, one_step_predictors)
        self.control_range = control_range

        assert isinstance(control_range[0], Sequence), "control_range must be a sequence of sequences"
        assert len(control_range[0]) == 2, "control_range must be a sequence of sequences of length 2"
        assert len(control_range) == system_type.control_size, \
            "control_size must be equal to the length of control_range"
        assert control_range[0][0] < control_range[0][1], "control_range must a valid range"

        self.smoothing = smoothing

        self.last_input = None
        self.smoothing_value = None

    def reset(self, init_idx: Sequence[int] = None, batch_size: int = None):
        super().reset(init_idx, batch_size)
        self.last_input = None
        # If smoothing is -1 random sample a smoothing value between 0 and 0.8
        # If smoothing is None, set smoothing to 0
        # Otherwise set smoothing to the given value
        if self.smoothing == -1:
            self.smoothing_value = random.random() * 0.8
        elif self.smoothing is None:
            self.smoothing_value = 0
        else:
            self.smoothing_value = self.smoothing

    def obtain_sol(self,
                   curr_s: Float[Tensor, "batch state"],
                   goal_trajectory: Float[Tensor, "batch time state"],
                   total_step: int,
                   skip: bool = False) -> Float[Tensor, "batch control"]:
        """
        Calculates the control input for the current state (random control)

        Args:
            curr_s: Current state; shape(batch, state_size)
            goal_trajectory: Goal trajectory; shape(batch, time, state_size)
            total_step: Current step in the episode

        Returns:
            u: Control input; shape(batch, control_size)
        """
        assert self.reseted, "You must call reset before calling obtain_sol"
        assert skip is False, "skip is not supported for RandomController"

        random_noise = torch.rand((curr_s.size(0), self.system_type.control_size), device=curr_s.device)
        for i in range(self.system_type.control_size):
            random_noise[:, i] *= (self.control_range[i][1] - self.control_range[i][0])
            random_noise[:, i] += self.control_range[i][0]

        if self.last_input is not None:
            random_noise = self.smoothing_value * self.last_input + (1 - self.smoothing_value) * random_noise
        self.last_input = random_noise
        return random_noise

    def __str__(self):
        return "RandomController"
