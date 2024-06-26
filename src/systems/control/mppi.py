from logging import getLogger
from typing import List, Union, Sequence

import torch
from jaxtyping import Float
from torch import Tensor, tensor
from torch.distributions import Normal
from torch.types import Device

from src.systems.control.controller import Controller
from src.systems.enviroment.env import calc_cost, StateCost, \
    TerminalCost, InputCost
from src.systems.predictor.moving_window_nn_predictor import MovingWindowNNPredictor
from src.systems.predictor.predictor import Predictor
from src.systems.systems.dynamic_system import DynamicSystemType

logger = getLogger(__name__)


class MPPI(Controller):
    """ Model Predictive Path Integral for linear and nonlinear method

    Ref:
        Nagabandi, A., Konoglie, K., Levine, S., & Kumar, V. (2019).
        Deep Dynamics Models for Learning Dexterous Manipulation.
        arXiv preprint arXiv:1909.11652.
    """

    def __init__(self,
                 predictor: Predictor,
                 pred_len: int,
                 beta: float,
                 popsize: int,
                 kappa: float,
                 noise_sigma: float,
                 system_type: DynamicSystemType,
                 state_cost: Union[StateCost, None],
                 terminal_state_cost: Union[TerminalCost, None],
                 input_cost: Union[InputCost, None],
                 input_bounds: Union[Float[Tensor, "input_size 2"], Sequence[Sequence[float]]],
                 log_all_unrolls: bool = False,
                 iterations: int = 1,
                 particles: int = 1,
                 multiply_prob: bool = False,
                 one_step_predictors: List[Predictor] = None,
                 noise_factor: float = 1.0,):
        """
        Args:
            predictor: The dynamic system model to predict the trajectory
            pred_len: Prediction horizon
            beta:
            popsize: Number of trajectories to sample
            kappa:
            noise_sigma:
            state_cost_fn:
            terminal_state_cost_fn:
            input_cost_fn:
            input_bounds: Bounds for the control input; shape(input_size, 2)
        """
        super(MPPI, self).__init__(predictor, system_type, one_step_predictors, state_cost, terminal_state_cost,
                                   input_cost)

        # model
        self.predictor = predictor

        # general parameters
        self.pred_len = pred_len
        self.input_size = self.system_type.control_size

        # mppi parameters
        self.beta = beta
        self.pop_size = popsize
        self.kappa = kappa
        self.noise_sigma = noise_sigma
        self.opt_dim = self.input_size * self.pred_len

        self.particles = particles
        self.multiply_prob = multiply_prob

        self.noise_factor = noise_factor

        # get bound
        if isinstance(input_bounds, Sequence):
            input_bounds = tensor(input_bounds, device=self.device)
        self.input_bounds = input_bounds
        self.input_upper_bounds = torch.tile(self.input_bounds[:, 1], (self.pred_len, 1))
        self.input_lower_bounds = torch.tile(self.input_bounds[:, 0], (self.pred_len, 1))

        ##Init with assumed batchsize = 1
        # init mean
        self.prev_sol = (self.input_upper_bounds[None] + self.input_lower_bounds[None]) / 2
        self.prev_sol = self.prev_sol.reshape(1, self.pred_len, self.input_size)

        # save
        self.last_u = torch.zeros((1, self.input_size), device=self.device)

        self.log_all_unrolls = log_all_unrolls
        if self.log_all_unrolls:
            self.unrolls = []
            self.costs = []
            self.weights = []
            self.pred_unrolls = []

        self.iterations = iterations

    def to(self, device: Device):
        super().to(device)
        self.prev_sol = self.prev_sol.to(device)
        self.last_u = self.last_u.to(device)
        self.input_lower_bounds = self.input_lower_bounds.to(device)
        self.input_upper_bounds = self.input_upper_bounds.to(device)
        self.input_bounds = self.input_bounds.to(device)

    def obtain_sol(self,
                   curr_s: Float[Tensor, "batch state"],
                   goal_trajectory: Float[Tensor, "batch time state"],
                   total_step: int,
                   skip: bool = False) -> \
            Float[Tensor, "batch control"]:
        """
        calculate the optimal inputs using MPPI
        Args:
            curr_s: current state; shape(state_size)
            goal_trajectory: goal trajectory; shape(pred_len, state_size)
        Returns:
            sol: optimal input; shape(control_size)
        """
        # if there is no history for the predictor return zero as control
        if total_step < 2 or skip:
            if self.log_all_unrolls:
                self.weights.append(None)
                self.pred_unrolls.append(None)

            return torch.zeros((curr_s.size(0), self.input_size), device=self.device)
        with torch.no_grad():
            # get noised inputs
            local_noise_value = self.noise_sigma
            for _ in range(self.iterations):
                noise = torch.randn((curr_s.size(0), self.pop_size, self.pred_len,
                                     self.input_size), device=self.device) * local_noise_value

                for t in range(self.pred_len):
                    if t > 0:
                        noise[:, :, t, :] = self.beta * noise[:, :, t, :] + (1 - self.beta) * noise[:, :, t - 1, :]
                    else:
                        noise[:, :, t, :] = self.beta * noise[:, :, t, :]

                noised_inputs = self.prev_sol[:, None, :, :] + noise

                # clip actions
                noised_inputs = torch.clip(
                    noised_inputs, self.input_lower_bounds, self.input_upper_bounds)

                # create particle copies
                noised_inputs = noised_inputs.repeat(1, self.particles, 1, 1)

                # calc cost
                costs, latent_prob = self._calc_cost(curr_s, noised_inputs, goal_trajectory)
                rewards = -costs

                # mppi update
                # normalize and get sum of reward
                # exp_rewards.shape = (N, )
                exp_rewards = torch.exp(self.kappa * (rewards - torch.max(rewards, dim=1).values[:, None])) * latent_prob
                denom = torch.sum(exp_rewards, dim=1) + torch.finfo(torch.float32).eps # avoid division by zero

                # weight actions
                weighted_inputs = exp_rewards[:, :, None, None] * noised_inputs
                sol = torch.sum(weighted_inputs, dim=1) / denom[:, None, None]

                if self.log_all_unrolls:
                    self.weights.append(exp_rewards / denom)
                    # TODO: Extend to multiple batches and remove assertion in mppi and random shooting
                    pred_unroll, _ = self.predictor.predict_mean_std_one_batch(curr_s, control=sol, batch_nr=0)
                    self.pred_unrolls.append(pred_unroll[0])

                # update prev sol for next inner mppi iteration
                self.prev_sol[:, :] = sol[:, :]
                # Update noise to converge to the optimal solution
                local_noise_value *= self.noise_factor

            # update prev sol for next env step
            self.prev_sol[:, :-1] = sol[:, 1:]
            self.prev_sol[:, -1] = sol[:, -1]  # last use the terminal input (repeat the last input)

            # render_unroll = False
            # if render_unroll:
            #     pred_unroll, _ = self.predictor.predict_mean_std_one_batch(curr_s[:1], control=sol[:1], batch_nr=0)
            #
            #     model = HalfCheetahEnv(render_mode="human")
            #     #model = InvertedPendulumEnv(render_mode="human")
            #     model.reset()
            #     model.render()
            #     print("Start rendering")
            #     for i in range(60):
            #         k = input("Press a key to cotinue")
            #         if k == "c":
            #             break
            #         # direcly update memory of model and data with env.history_model[i] and env.history_data[i]
            #         model.set_state(pred_unroll[0, i%30].cpu().numpy()[:9], pred_unroll[0, i%30].cpu().numpy()[9:])
            #         #model.set_state(pred_unroll[0, i%30].cpu().numpy()[:2], pred_unroll[0, i%30].cpu().numpy()[2:])
            #         model.render()
            # log
            self.last_u = sol[:, 0]
            control = sol[:, 0]

        learn_flag = self.predictor.get_learn_flag()
        control[learn_flag] = (torch.rand((learn_flag.sum(), self.input_size), device=self.device)
                               * (self.input_upper_bounds[0] - self.input_lower_bounds[0]) + self.input_lower_bounds[0])

        return control

    def _calc_cost(self,
                   curr_s: Float[Tensor, "batch state"],
                   control_samples: Float[Tensor, "batch sample time control"],
                   goal_trajectory: Float[Tensor, "batch time state"]) -> \
            Union[Float[Tensor, "batch sample"], Float[Tensor, "batch sample"]]:
        """

        Args:
            curr_s:  Current state; shape (batch, state_size)
            control_samples: Input samples; shape (batch,sample, pred_len, input_size)
            goal_trajectory: Goal trajectory; shape (batch, pred_len, state_size)

        Returns:
            Cost for each sample; shape (sample)
        """
        # get size
        pop_size = control_samples.size(1)
        state_size = curr_s.size(1)
        batch_size = curr_s.size(0)

        latent_z = None
        z_dist: Normal = None

        if isinstance(self.predictor, MovingWindowNNPredictor):
            batched_curr_s = curr_s[:, None, :].repeat(1, pop_size, 1).reshape(-1, state_size)
            batched_control_samples = control_samples.reshape(-1, control_samples.size(-2),
                                                              control_samples.size(-1))
            batched_pred_xs, batched_pred_xs_std, latent_z, z_dist = self.predictor.predict_mean_std_z_all_batch(
                batched_curr_s, control=batched_control_samples, pop_size=pop_size)
            batched_pred_xs = batched_pred_xs.reshape(batch_size, pop_size, self.pred_len, state_size)
            batched_pred_xs_std = batched_pred_xs_std.reshape(batch_size, pop_size, self.pred_len, state_size)
        else:
            batched_pred_xs = torch.zeros((batch_size, pop_size, control_samples.size(-2), curr_s.size(1)),
                                          device=self.device)
            batched_pred_xs_std = torch.zeros((batch_size, pop_size, self.pred_len, curr_s.size(1)), device=self.device)
            for batch in range(batch_size):
                # calc cost, pred_xs.shape = (pop_size, pred_len+1, state_size)
                batched_curr_s = curr_s[batch, None, :].repeat(pop_size, 1).reshape(-1, state_size)
                batched_control_samples = control_samples[batch].reshape(-1, control_samples.size(-2),
                                                                         control_samples.size(-1))
                pred_xs, pred_xs_std = self.predictor.predict_mean_std_one_batch(
                    batched_curr_s, control=batched_control_samples, batch_nr=batch)
                batched_pred_xs[batch] = pred_xs
                batched_pred_xs_std[batch] = pred_xs_std

        if latent_z is not None and self.multiply_prob:
            latent_z = latent_z.reshape(batch_size, pop_size, -1).permute(1, 0, 2)
            latent_prob = torch.exp(z_dist.log_prob(latent_z).sum(dim=-1)).permute(1, 0)
        else:
            latent_prob = torch.ones((batch_size, pop_size), device=self.device)

        # get particle cost
        costs = calc_cost(batched_pred_xs, control_samples, goal_trajectory,
                          self.state_cost_fn, self.input_cost_fn,
                          self.terminal_state_cost_fn)

        if self.log_all_unrolls:
            self.unrolls.append(batched_pred_xs)
            self.costs.append(costs)

        return costs, latent_prob

    def reset(self, init_idx: List[int], batch_size: int):
        super().reset(init_idx, batch_size)
        self._reset(batch_size)

    def reset_with_given_data(self,
                              init_state: Float[Tensor, "batch state"],
                              parameters: Float[Tensor, "batch parameter"]):
        super().reset_with_given_data(init_state, parameters)
        self._reset(init_state.size(0))

    def _reset(self, batch_size: int):
        if self.log_all_unrolls:
            assert batch_size == 1, "batch_size must be 1 when logging all unrolls"
            # init mean
        self.prev_sol = (self.input_upper_bounds[None] + self.input_lower_bounds[None]) / 2
        self.prev_sol = self.prev_sol.tile((batch_size, 1, 1))
        self.prev_sol = self.prev_sol.reshape(batch_size, self.pred_len, self.input_size)

        # save
        self.last_u = torch.zeros((batch_size, self.input_size), device=self.device)

    def __str__(self):
        return "MPPI"
