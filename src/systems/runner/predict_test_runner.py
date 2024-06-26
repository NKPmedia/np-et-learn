from dataclasses import dataclass
from logging import getLogger
from typing import List

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor
from typeguard import typechecked

from src.systems.predictor.predictor import Predictor
from src.systems.systems import NNModel
from src.systems.systems.dynamic_system import DynamicSystem

logger = getLogger(__name__)


@dataclass(init=True)
class SystemResult:
    system_states: Float[Tensor, "time state"]
    observed_states: Float[Tensor, "time state"]
    controls: Float[Tensor, "time control"]


@dataclass(init=True)
class PredictErrorResult:
    toObserved: Float[Tensor, "time state"]
    toSystem: Float[Tensor, "time state"]


@dataclass(init=True)
class PredictionResult:
    system_states: Float[Tensor, "time state"]
    std: Float[Tensor, "time state"]
    trigger: Float[Tensor, "time"]
    window_size: Float[Tensor, "time"]
    error: PredictErrorResult
    latent_std: Float[Tensor, "time hidden_state"]
    all_system_states: Float[Tensor, "model time state"]
    cross_attention_weights: List[Float[Tensor, "state"]] = None


@dataclass(init=True)
class ControlResult:
    cost: Float[Tensor, "time"]
    score: Float[Tensor, "1"]


@dataclass(init=True)
class EnvRunResult:
    delta_t: float
    system: SystemResult
    prediction: PredictionResult
    control: ControlResult = None

@dataclass(init=True)
class ShortEnvRunResult:
    delta_t: float
    score: Float[Tensor, "1"]

@dataclass(init=True)
class ShortPredEnvRunResult:
    delta_t: float
    score: Float[Tensor, "1"]
    errorToSystem: Float[Tensor, "time state"]
    std: Float[Tensor, "time state"]
    cost: Float[Tensor, "time"]

class PredictTestRunner():
    """
    experiment runner
    """

    def __init__(self, env, predictor: Predictor, parallel_runs: int, device: str = "cpu",
                 print_steps: bool = False) -> None:
        """
        Args:
            env: The simulation environment and dynamics (Also defines the goal)
            predictor: The predictor that predicts the next state
            device: The device to run the experiment on (cpu or cuda)
            print_steps: If true, prints the current step
        """
        self.env = env
        self.predictor = predictor
        self.device = device
        self.print_steps = print_steps
        self.parallel_runs = parallel_runs
        assert device in ["cpu", "cuda"]
        self.device = torch.device(device)
        env.to(self.device)
        predictor.to(self.device)

    def run(self, init_idx: List[int]) -> \
            List[EnvRunResult]:
        """
        Runs one episode of the experiment given an environment, predictor
        and calculates the error
        Args:
            env: The simulation environment and dynamics (Also defines the goal)
            predictor: The predictor that predicts the next state
            device: The device to run the experiment on (cpu or cuda)
            print_steps: If true, prints the current step
        Returns:
            res (EnvRunResult): The results of the experiment

        """
        done = False
        curr_y = self.env.reset(init_idx=init_idx)
        self.predictor.reset(init_idx=init_idx)
        history_y, history_s, history_prediction, history_errory, history_errors, history_u, history_std \
            = [], [], [], [], [], [], []
        history_latent_std = []
        step_count = 0
        score = 0.

        self.predictor.set_state(
            torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device),
            torch.empty((curr_y.size(0), 0, self.env.control_size), device=self.device),
            torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device),
            torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device))

        while not done:
            if (step_count % 50) == 0 and self.print_steps:
                print(f"Step = {step_count}")
            logger.debug("Step = {}".format(step_count))
            # predict

            control = self.env.get_last_control()
            predicted_sol, predict_std = self.predictor.next_state(curr_y, control, step_count)
            latent_std = self.predictor.get_latent_std(curr_y, control, step_count)

            # step
            next_y, cost, done = self.env.step(control, step_count)
            next_u = self.env.get_last_control()
            next_s = self.env.curr_s

            error_y = torch.abs(predicted_sol - next_y)
            error_s = torch.abs(predicted_sol - next_s)

            # save
            history_y.append(next_y)
            history_s.append(next_s)
            history_prediction.append(predicted_sol)
            history_std.append(predict_std)
            history_errory.append(error_y)
            history_errors.append(error_s)
            history_u.append(next_u)
            history_latent_std.append(latent_std)
            # update
            curr_y = next_y
            score += cost

            self.predictor.set_state(
                torch.stack(history_y, dim=1),
                torch.stack(history_u, dim=1),
                torch.stack(history_prediction, dim=1),
                torch.stack(history_std, dim=1)) \
                if len(history_y) > 0 else self.predictor.set_state(
                torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device),
                torch.empty((curr_y.size(0), 0, self.env.control_size), device=self.device),
                torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device),
                torch.empty((curr_y.size(0), 0, curr_y.size(1)), device=self.device))

            step_count += 1

        history_y = torch.stack(history_y, dim=1)
        history_s = torch.stack(history_s, dim=1)
        history_prediction = torch.stack(history_prediction, dim=1)
        history_std = torch.stack(history_std, dim=1)
        history_errory = torch.stack(history_errory, dim=1)
        history_errors = torch.stack(history_errors, dim=1)
        history_u = torch.stack(history_u, dim=1)
        history_latent_std = torch.stack(history_latent_std, dim=1)

        res = [EnvRunResult(
            delta_t=self.env.dt,
            system=SystemResult(
                system_states=history_s[i].detach().cpu(),
                observed_states=history_y[i].detach().cpu(),
                controls=history_u[i].detach().cpu()
            ),
            prediction=PredictionResult(
                system_states=history_prediction[i].detach().cpu(),
                all_system_states=None,
                std=history_std[i].detach().cpu(),
                trigger=self.predictor.trigger[i].detach().cpu(),
                window_size=self.predictor.actual_window_size[i].detach().cpu()
                , error=PredictErrorResult(
                    toObserved=history_errory[i].detach().cpu(),
                    toSystem=history_errors[i].detach().cpu()
                ),
                latent_std=history_latent_std[i].detach().cpu()
            )
        ) for i in range(self.parallel_runs)]

        return res
