from typing import Union, Sequence, List, Mapping, Dict

import torch
from jaxtyping import Float
from torch import Tensor
from torch.types import Device

from src.systems.systems.dynamic_system import DynamicSystemType
from src.systems.enviroment.env import StateCost, TerminalCost, InputCost
from src.systems.predictor.predictor import Predictor


class Controller:

    def __init__(self,
                 predictor: Predictor,
                 system_type: DynamicSystemType,
                 one_step_predictors: List[Predictor] = None,
                 state_cost: Union[StateCost, None] = None,
                 terminal_state_cost: Union[TerminalCost, None] = None,
                 input_cost: Union[InputCost, None] = None):
        self.predictor = predictor

        if one_step_predictors is None:
            self.different_one_step_predictor = False
            self.one_step_predictors = [predictor]
        else:
            self.different_one_step_predictor = True
            self.one_step_predictors = one_step_predictors
            if isinstance(self.one_step_predictors, Mapping):
                self.one_step_predictors = list(self.one_step_predictors.values())
        self.system_type = system_type

        # get cost func
        self.state_cost_fn = state_cost.cost if state_cost is not None else None
        self.terminal_state_cost_fn = terminal_state_cost.cost if terminal_state_cost is not None else None
        self.input_cost_fn = input_cost.cost if input_cost is not None else None
        self.device = None

        self.reseted = False

    def to(self, device: Device):
        self.device = device
        self.predictor.to(device)
        if self.different_one_step_predictor:
            for predictor in self.one_step_predictors:
                predictor.to(device)

    def obtain_sol(self,
                   curr_s: Float[Tensor, "batch state"],
                   goal_trajectory: Float[Tensor, "batch time state"],
                   total_step: int,
                   skip: bool) -> \
            Float[Tensor, "batch control"]:
        raise NotImplementedError("Implement the algorithm to \
                                   get optimal input")

    def set_state(self, history_y: Float[Tensor, "batch time state"],
                  history_u: Float[Tensor, "batch time control"],
                  history_prediction: Float[Tensor, "batch time state"],
                  history_std: Float[Tensor, "batch time state"]):
        self.predictor.set_state(history_y, history_u, history_prediction, history_std)
        if self.different_one_step_predictor:
            for predictor in self.one_step_predictors:
                predictor.set_state(history_y, history_u, history_prediction, history_std)

    def predict_next_state(self, curr_y, control, step, skip: bool = False):
        preds = []
        stds = []
        for predictor in self.one_step_predictors:
            pred, std = predictor.next_state(curr_y, control, step, skip)
            preds.append(pred)
            stds.append(std)
        return torch.stack(preds), torch.stack(stds)

    def reset(self, init_idx: Sequence[int], batch_size: int):
        self.reseted = True
        self.predictor.reset(init_idx)
        if self.different_one_step_predictor:
            for predictor in self.one_step_predictors:
                predictor.reset(init_idx)

    def reset_with_given_data(self,
                              init_state: Float[Tensor, "batch state"],
                              parameters: Float[Tensor, "batch parameter"]):
        self.reseted = True
        self.predictor.reset_with_given_data(init_state, parameters)
        if self.different_one_step_predictor:
            for predictor in self.one_step_predictors:
                predictor.reset_with_given_data(init_state, parameters)

    def set_model(self, model, one_step_models=None):
        assert one_step_models is None or self.different_one_step_predictor, \
            "only give a one step model if you have a different one step predictor"
        self.predictor.set_model(model)
        if self.different_one_step_predictor:
            for one_step_predictor, one_step_model in zip(self.one_step_predictors, one_step_models):
                one_step_predictor.set_model(one_step_model)

    @property
    def trigger(self) -> Float[Tensor, "batch time"]:
        return self.predictor.trigger

    @property
    def actual_window_size(self) -> Float[Tensor, "batch time"]:
        return self.predictor.actual_window_size

    def get_latent_std(self, curr_y, control, step: int, skip: bool = False) -> Float[Tensor, "batch hidden_size"]:
        return self.predictor.get_latent_std(curr_y, control, step, skip)

    def get_attentions(self, curr_y, control, step_count) -> Dict:
        return self.predictor.get_attentions(curr_y, control, step_count)

    def add_interaction_idx(self, step_count):
        self.predictor.add_interaction_idx(step_count)
        if self.different_one_step_predictor:
            for predictor in self.one_step_predictors:
                predictor.add_interaction_idx(step_count)