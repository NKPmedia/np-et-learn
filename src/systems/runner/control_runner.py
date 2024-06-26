from logging import getLogger
from typing import List, Sequence, Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from tqdm import tqdm

from src.data.parameter_changer import ParameterChanger
from src.systems.control.controller import Controller
from src.systems.planner.planner import Planner
from src.systems.runner.interactors import Interactor
from src.systems.runner.predict_test_runner import EnvRunResult, SystemResult, PredictionResult, PredictErrorResult, \
    ControlResult

logger = getLogger(__name__)


class ControlRunner():
    """
    experiment runner
    """

    def __init__(self,
                 env,
                 controller: Controller,
                 planner: Planner,
                 device: str = "cpu",
                 print_steps: bool = False,
                 do_predictions: bool = True,
                 do_latent_predictions: bool = True,
                 control_interactor: Interactor = Interactor(),
                 state_interactor: Interactor = Interactor(),
                 log_attention: bool = False) -> None:
        """
        Args:
            env: The simulation environment and dynamics (Also defines the goal)
            predictor: The predictor that predicts the next state
            device: The device to run the experiment on (cpu or cuda)
            print_steps: If true, prints the current step
        """
        self.env = env
        self.controller = controller
        self.device = device
        self.print_steps = print_steps
        self.planner = planner
        self.control_interactor = control_interactor
        self.state_interactor = state_interactor
        self.log_attention = log_attention
        assert device in ["cpu", "cuda"]
        self.device = torch.device(device)
        env.to(self.device)
        controller.to(self.device)
        control_interactor.to(self.device)
        state_interactor.to(self.device)

        self.curr_y = None
        self.done = False

        self.do_predictions = do_predictions
        self.do_latent_predictions = do_latent_predictions

    def _reset_to_random(self, parameter_changer):
        with torch.no_grad():
            self.done = False
            self.curr_y = self.env.reset(parameter_changer=parameter_changer)
            self.controller.reset(list(range(self.env.batch_size)), self.env.batch_size)
            if isinstance(self.controller.predictor, ExactEnvPredictor):
                self.controller.predictor.reset_with_parameterchanger(self.env.batch_size, parameter_changer)
        self._reset_history()
        self.control_interactor.reset(batch_size=self.env.batch_size, channel_size=self.controller.system_type.control_size)
        self.state_interactor.reset(batch_size=self.env.batch_size, channel_size=self.controller.system_type.state_size)

    def _reset_to_idx(self, init_idx: Sequence[int]):
        with torch.no_grad():
            self.done = False
            self.curr_y = self.env.reset(init_idx=init_idx)
            self.controller.reset(init_idx=init_idx, batch_size=len(init_idx))
        self._reset_history()
        self.control_interactor.reset(batch_size=len(init_idx), channel_size=self.controller.system_type.control_size)
        self.state_interactor.reset(batch_size=len(init_idx), channel_size=self.controller.system_type.state_size)

    def _reset_history(self):
        with torch.no_grad():
            assert self.curr_y is not None, "Please call reset() first"
            self.history_y = []
            self.history_s = []
            self.history_prediction = []
            self.history_predictions = []
            self.history_errory = []
            self.history_errors = []
            self.history_u = []
            self.history_std = []
            self.history_latent_std = []
            self.history_cost = []
            self.history_cross_attention = []

            self.score = None

            self._update_controller_state()

    def _update_controller_state(self):
        self.controller.set_state(
            torch.stack(self.history_y, dim=1),
            torch.stack(self.history_u, dim=1),
            torch.stack(self.history_prediction, dim=1),
            torch.stack(self.history_std, dim=1)) \
            if len(self.history_y) > 0 else self.controller.set_state(
            torch.empty((self.curr_y.size(0), 0, self.curr_y.size(1)), device=self.device),
            torch.empty((self.curr_y.size(0), 0, self.controller.system_type.control_size), device=self.device),
            torch.empty((self.curr_y.size(0), 0, self.curr_y.size(1)), device=self.device),
            torch.empty((self.curr_y.size(0), 0, self.curr_y.size(1)), device=self.device))

    def _run_loop(self):
        with torch.no_grad():
            goal = self.env.get_goal()
            bar = tqdm(total=self.env.max_step)
            step_count = 0
            done = False
            while not done:
                bar.update()
                if (step_count % 50) == 0 and self.print_steps:
                    print(f"Step = {step_count}")
                logger.debug("Step = {}".format(step_count))

                #
                skip_pred_and_control = step_count-1 in self.controller.predictor.interaction_idx
                g_xs = self.planner.plan(self.curr_y, goal)

                control = self.controller.obtain_sol(self.curr_y, g_xs, total_step=step_count, skip=skip_pred_and_control)
                # apply external control disturbance
                interaction = self.control_interactor.get_interactions(step_count)
                #check if interaction is has one non zero element
                if interaction.sum() != 0:
                    control = interaction

                if len(self.history_u) > 0:
                    self.history_u[-1] = control

                if self.do_predictions:
                    predicts_sol, predicts_std = self.controller.predict_next_state(self.curr_y, control, step_count, skip=skip_pred_and_control)
                    predicted_sol = predicts_sol[0]
                    predict_std = predicts_std[0]
                if self.log_attention:
                    c_attentions = self.controller.get_attentions(self.curr_y, control, step_count)
                    self.history_cross_attention.append(c_attentions)

                if self.do_latent_predictions:
                    latent_std = self.controller.get_latent_std(self.curr_y, control, step_count, skip=skip_pred_and_control)


                # check if interaction is has one non zero element
                state_interaction = self.state_interactor.get_interactions(step_count)
                if state_interaction.sum() != 0:
                    self.env.overwrite_state(state_interaction)
                    self.controller.add_interaction_idx(step_count)
                # step
                next_y, cost, done = self.env.step(control, step_count)
                next_s = self.env.curr_s

                if self.do_predictions:
                    error_y = torch.abs(predicted_sol - next_y)
                    error_s = torch.abs(predicted_sol - next_s)

                    self.history_errory.append(error_y)
                    self.history_errors.append(error_s)
                    self.history_prediction.append(predicted_sol)
                    self.history_std.append(predict_std)
                    self.history_predictions.append(predicts_sol)

                # save
                self.history_y.append(next_y)
                self.history_s.append(next_s)

                if self.do_latent_predictions:
                    self.history_latent_std.append(latent_std)

                self.history_u.append(torch.zeros_like(control))  # We don't have the control for the next step yet
                self.history_cost.append(cost)
                # update
                self.curr_y = next_y

                self.score = cost if (self.score is None) else self.score + cost

                self._update_controller_state()

                step_count += 1

        self.env.close_video_recorder()

    def _convert_res_to_tensor(self):
        self.history_y = torch.stack(self.history_y, dim=1)
        self.history_s = torch.stack(self.history_s, dim=1)
        self.history_prediction = torch.stack(self.history_prediction, dim=1)
        self.history_std = torch.stack(self.history_std, dim=1)
        self.history_errory = torch.stack(self.history_errory, dim=1)
        self.history_errors = torch.stack(self.history_errors, dim=1)
        self.history_u = torch.stack(self.history_u, dim=1)
        self.history_cost = torch.stack(self.history_cost, dim=1)
        self.history_latent_std = torch.stack(self.history_latent_std, dim=1)
        self.history_predictions = torch.stack(self.history_predictions, dim=2)
        self.history_cross_attention = [[
            [att[i, 0] for att in self.history_cross_attention]
        ] for i in range(self.history_y.size(0))] if self.log_attention else None

    def get_trajectories_by_random(self, parameter_changer: ParameterChanger) -> \
            Tuple[Float[Tensor, "batch time state"],
            Float[Tensor, "batch time control"],
            Float[Tensor, "batch time cost"]]:

        self._reset_to_random(parameter_changer)
        self._run_loop()
        self._convert_res_to_tensor()

        return self.history_y, self.history_u, self.history_cost

    def run(self, init_idx: List[int]) -> \
            List[EnvRunResult]:
        """
        Runs one episode of the experiment given an environment, predictor
        and calculates the error
        Args:
            init_idx: The idxs of the run that should be simulated
            env: The simulation environment and dynamics (Also defines the goal)
            predictor: The predictor that predicts the next state
            device: The device to run the experiment on (cpu or cuda)
            print_steps: If true, prints the current step
        Returns:
            res (EnvRunResult): The results of the experiment

        """

        self._reset_to_idx(init_idx)
        self._run_loop()
        self._convert_res_to_tensor()

        res = [EnvRunResult(
            delta_t=self.env.dt,
            system=SystemResult(
                system_states=self.history_s[i].detach().cpu(),
                observed_states=self.history_y[i].detach().cpu(),
                controls=self.history_u[i].detach().cpu()
            ),
            prediction=PredictionResult(
                system_states=self.history_prediction[i].detach().cpu(),
                all_system_states=self.history_predictions[:, i].detach().cpu(),
                std=self.history_std[i].detach().cpu(),
                trigger=self.controller.trigger[i].detach().cpu(),
                window_size=self.controller.actual_window_size[i].detach().cpu()
                , error=PredictErrorResult(
                    toObserved=self.history_errory[i].detach().cpu(),
                    toSystem=self.history_errors[i].detach().cpu()
                ),
                latent_std=self.history_latent_std[i].detach().cpu(),
                cross_attention_weights=self.history_cross_attention[i] if self.log_attention else None
            ),
            control=ControlResult(
                cost=self.history_cost[i].detach().cpu(),
                score=self.score[i].detach().cpu()
            )
        ) for i in range(len(init_idx))]

        return res
