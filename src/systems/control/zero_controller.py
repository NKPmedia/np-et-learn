from logging import getLogger
from typing import List

import torch
from jaxtyping import Float
from torch import Tensor

from .controller import Controller
from src.systems.systems.dynamic_system import DynamicSystem, DynamicSystemType
from ..predictor.predictor import Predictor

logger = getLogger(__name__)


class ZeroController(Controller):
    """ Zero controller. Only applies zero control input
    """
    def __init__(self, predictor: Predictor, system_type: DynamicSystemType, one_step_predictors: List[Predictor] = None):
        super(ZeroController, self).__init__(predictor, system_type, one_step_predictors)

    def clear_sol(self):
        pass

    def obtain_sol(self,
                   curr_s: Float[Tensor, "batch state"],
                   goal_trajectory: Float[Tensor, "batch time state"],
                   total_step: int,
                   skip: bool = False) -> Float[Tensor, "batch control"]:#
        return torch.zeros((curr_s.size(0), self.system_type.control_size), device=curr_s.device)

    def __str__(self):
        return "ZeroController"