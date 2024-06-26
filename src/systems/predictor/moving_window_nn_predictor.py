from types import NoneType
from typing import List, Tuple, Union

import torch
from jaxtyping import Float, Bool
from torch import Tensor

from src.systems.predictor.predictor import Predictor
from src.systems.predictor.reset.reset_trigger import ResetTrigger, ConstantTrigger
from src.systems.systems.NNModel import NNModel


class MovingWindowNNPredictor(Predictor):
    """
    Selects the context points from a set of trajectories using a moving window
    """

    def __init__(self, model: NNModel = None, window_size: int = 200,
                 reset_trigger: ResetTrigger = ConstantTrigger([]),
                 fill_small_windows: bool = False,
                 log_zeroControl_unroll: bool = False, unroll_mode: str = "mean_propagation"):
        """
        Args:
            window_size: size of the moving window
        """
        super().__init__(log_zeroControl_unroll)
        self.window_size = window_size
        self.reset_trigger = reset_trigger
        self.model = model
        self.window_starts = None
        self.trigger = None
        self.actual_window_size = None
        self.device = "cpu"
        self.unroll_mode = unroll_mode
        self.fill_small_windows = fill_small_windows
        assert self.unroll_mode in ["mean_propagation", "sampling", "auto_regressive_sampling"]

    def reset(self, init_idx: List[int] = None):
        """
        Needs to be called before a new trajectory is predicted

        Returns:

        """
        super().reset(init_idx)
        self.history_y = None
        self.history_u = None
        self.trigger = None
        self.window_starts = None
        self.actual_window_size = None
        self.reset_trigger.reset(len(init_idx))
        self.interaction_idx = []


    def to(self, device):
        self.model.to(device)
        self.reset_trigger.to(device)
        self.device = device

    def set_state(self,
                  history_y: Union[Float[Tensor, "batch time state"], NoneType],
                  history_u: Union[Float[Tensor, "batch time control"], NoneType],
                  predictions: Union[Float[Tensor, "batch time state"], NoneType],
                  predictions_std: Union[Float[Tensor, "batch time state"], NoneType]):
        """
        Saves the history of states
        should be called just ones every step in the simulation
        Args:
            history_y: History of observations; shape(batch, time, state_size)
            history_u: History of controls; shape(batch, time, control_size)
            predictions: Predictions of states; shape(batch, time, state_size)
            predictions_std: Standard deviation of the predictions; shape(batch, time, state_size)
        """
        super().set_state(history_y, history_u, predictions, predictions_std)
        history_len = history_y.size(1)
        if history_len != 0:  # Trigger evaluation start after the first state computation
            triggers = self._check_trigger(history_y.size(0))
            self._update_window_starts(triggers, history_len, self.window_size)
            self.reseted_idx = triggers.nonzero()[:, 0]

    def next_state(self, current_s: Float[Tensor, "batch state"], control: Float[Tensor, "batch control"],
                   step_count: int, skip: bool = False) -> \
            Tuple[Float[Tensor, "batch state"], Float[Tensor, "batch state"]]:
        """
        Predicts the next state using a neural network
        Sets the context based on a moving window

        If there are not enough points in the history, the next state is just the current state
        Args:
            control: Control; shape(batch control_size)
            current_s: Current state; shape(batch state_size)
        Returns:
            next_s: Next state; shape(batch state_size)
            next_s_std: Standard deviation of the next state; shape(batch state_size)
        """
        assert self.model is not None, "Model is not set"

        if self.history_y is not None and self.history_y.size(1) > 1 and not skip:
            context_x, context_y, context_sizes = self._get_context()

            self.model.set_context(context_x, context_y)
            next_s, next_std = self.model.next_state_with_std(
                current_s[:, None],
                control=control[:, None],
                context_sizes=context_sizes[:])
            next_s = next_s[:, 0]
            next_std = next_std[:, 0]
        else:
            next_s = current_s
            # Pred is likely very bad if there is no context
            next_std = torch.zeros_like(current_s) + 1
        if next_s.isnan().any():
            raise ValueError("Predicted state contains NaN")
        return next_s, next_std

    def predict_mean_std_one_batch(self,
                                   current_s: Float[Tensor, "sample state"],
                                   control: Float[Tensor, "sample time control"],
                                   batch_nr: int) -> \
            Tuple[Float[Tensor, "sample time state"], Float[Tensor, "sample time state"]]:
        """
        Unrolls the system for a given amount of time
        Assumes that all the samples are from the same batch nr (same system)
        Therefore we can use the same context for all the samples
        Args:
            current_s: Current state; shape(sample, state_size)
            control: Control; shape(sample, time, control_size)

        Returns:
            history_y: History of states; shape(sample, time, state_size)
            history_s_std: Standard deviation of the history of states; shape(sample, time, state_size)
        """
        assert self.model is not None, "Model is not set"
        if self.history_y is None or self.history_y.size(1) < 2:
            raise ValueError("History is not set or too short for a meaningfully unroll")

        context_x, context_y, context_sizes = self._get_context(batch_nr=batch_nr)

        self.model.set_context(context_x, context_y)
        next_ss, next_stds = self.model.predict_mean_std(current_s,
                                                         control,
                                                         context_sizes=context_sizes,
                                                         unroll_mode=self.unroll_mode)

        return next_ss, next_stds

    def predict_mean_std_z_all_batch(self, current_s, control, pop_size):
        context_x, context_y, context_sizes = self._get_context()

        self.model.set_context(context_x, context_y)
        next_s, next_std, z_latent, z_dist = self.model.predict_mean_std_z(
            current_s,
            control=control,
            context_sizes=context_sizes,
            pop_size=pop_size,
            unroll_mode=self.unroll_mode)
        return next_s, next_std, z_latent, z_dist

    def calc_log_zeroControl_unrolls(self):
        """
        Logs the unrolls where the control was zero
        """

        pred_length = 100
        pop_size = 20

        batch_size = self.history_y.size(0)
        control_size = self.history_u.size(2)
        state_size = self.history_y.size(2)

        # Checks if there are at least two states in the history (this is needed for nearly all prediction methods)
        if self.history_y.size(1) > 1:
            curr_y = self.history_y[:, -1, :]

            control = torch.zeros((pop_size * batch_size, pred_length, control_size), device=self.device)

            batched_curr_s = curr_y[:, None, :].repeat(1, pop_size, 1).reshape(-1, state_size)
            batched_pred_xs, batched_pred_xs_std = self.predict_mean_std_all_batch(
                batched_curr_s, control=control, pop_size=pop_size)
            batched_pred_xs = batched_pred_xs.reshape(batch_size, pop_size, pred_length, state_size)

            self.zeroControl_unrolls.append(batched_pred_xs.cpu().detach())

        else:
            self.zeroControl_unrolls.append(torch.zeros((batch_size, pop_size, pred_length, state_size)).cpu().detach())

    def _update_window_starts(self, trigger: Bool[Tensor, "batch"], history_len: int, window_size: int):
        """
        s_init -> s0 -> s1 -> s2 -> s3 -c> s4 -> s5
        Assume a change from s3 to s4 then the change is detected at step 4
        The new window should start at step 3
        At this point 5 states are in the history. contextx should be 3:4 and contexty 4:5.


        We computed step 4 and we trigger because s4 was weird.
        Therefore, the window starts at 3 (x - 1)

        Because step_count was computed

        Args:
            trigger:
            history_len:
            window_size:

        Returns:

        """
        step = history_len - 1
        if self.window_starts is None:
            self.window_starts = torch.zeros_like(trigger, dtype=torch.long)
        self.window_starts[self.window_starts < (step - window_size)] = step - window_size
        self.window_starts[trigger] = step - 1

        current_window_size = step - self.window_starts
        if self.actual_window_size is None:
            self.actual_window_size = current_window_size[:, None]
        else:
            self.actual_window_size = torch.concatenate([self.actual_window_size, current_window_size[:, None]], dim=1)

    def _get_context(self, batch_nr: int = None) -> \
            Tuple[Float[Tensor, "batch context_size state_control"],
            Float[Tensor, "batch context_size state"],
            Float[Tensor, "batch"]]:
        """
        Selects the context points from trajectories based on a moving window of selectable length
        Window is always at the end of the trajectory
        the windows starts at the point specified by window_starts
        The windows can have different lengths but the context gets padded with zeros

        Args:
            batch_nr: Batch number of the context to select; if None then all the batches are selected

        Returns:
            context_x: set of states; shape(batch context_size, state_size+control_size)
            context_y: set of next_states; shape(batch context_size, state_size+control_size)
            context_sizes: list of context sizes; shape(batch)
        """
        history = torch.concatenate([self.history_y, self.history_u], dim=-1)
        step = history.size(1) - 1
        if self.history_y.size(1) < 2:
            raise ValueError("Trajectory is too short for moving window context selector")

        contexts_x = []
        contexts_y = []
        context_sizes = []

        end_index = history.size(1) - 1  # (-1) Because the last state is part of the y context
        barch_range = range(history.size(0)) if batch_nr is None else [batch_nr]
        for batch in barch_range:
            start_index = self.window_starts[batch]
            context_x = history[batch, start_index:end_index]
            context_y = self.history_y[batch, start_index + 1:end_index + 1]
            allowed_idxes = self._get_allowed_context_idxes(start_index, end_index)

            context_x = context_x[allowed_idxes]
            context_y = context_y[allowed_idxes]

            if self.fill_small_windows and context_x.size(0) < self.window_size:
                context_x, context_y = self._fill_small_context(context_x, context_y,
                                                                      self.window_size)
            contexts_x.append(context_x)
            contexts_y.append(context_y)
            context_sizes.append(context_x.size(0))
            if context_sizes[-1] == 0:
                raise ValueError("Window size is 0. Not good for NPs Error")

        contexts_x, contexts_y = self._pad_contexts(contexts_x, contexts_y)

        return contexts_x, contexts_y, torch.tensor(context_sizes)

    def _check_trigger(self, batch_dim: int) -> Bool[Tensor, "batch"]:
        if self.history_y.size(1) > 0:
            step = self.history_y.size(1) - 1
            triggers = self.reset_trigger.test_trigger(
                self.predictions,
                self.predictions_std,
                self.history_y,
                step
            ).to(self.model.device)
        else:
            triggers = torch.zeros(batch_dim, dtype=torch.bool, device=self.model.device)
        if self.trigger is None:
            self.trigger = triggers[:, None]
        else:
            self.trigger = torch.concatenate([self.trigger, triggers[:, None]], dim=-1)
        return triggers

    def _pad_contexts(self,
                      contexts_x: List[Float[Tensor, "time state_control"]],
                      contexts_y: List[Float[Tensor, "time state_control"]]) -> \
            Tuple[Float[Tensor, "batch context_size state_control"], Float[Tensor, "batch context_size state"]]:
        """Pads the contexts with zeros such that they have the same length"""
        max_context_size = max([c.size(0) for c in contexts_x])
        contexts_x = [
            torch.cat([c, torch.zeros(max_context_size - c.size(0), c.size(1), device=contexts_y[0].device)], dim=0) for
            c in contexts_x]
        contexts_y = [
            torch.cat([c, torch.zeros(max_context_size - c.size(0), c.size(1), device=contexts_y[0].device)], dim=0) for
            c in contexts_y]
        contexts_x = torch.stack(contexts_x, dim=0)
        contexts_y = torch.stack(contexts_y, dim=0)
        return contexts_x, contexts_y

    def predict_mean_std_all_batch(self,
                                   current_s: Float[Tensor, "batch_pop_size control_size"],
                                   control: Float[Tensor, "batch_pop_size time control_size"],
                                   pop_size: int):

        context_x, context_y, context_sizes = self._get_context()

        self.model.set_context(context_x, context_y)
        next_s, next_std = self.model.predict_mean_std(
            current_s,
            control=control,
            context_sizes=context_sizes,
            pop_size=pop_size,
            unroll_mode=self.unroll_mode)
        return next_s, next_std

    def get_latent_std(self, current_s, control, step: int, skip: bool = False) -> Float[Tensor, "batch hidden_size"]:
        """
        Returns the standard deviation of the latent space
        Args:
            step:

        Returns:

        """
        assert self.model is not None, "Model is not set"

        if self.history_y.size(1) > 1 and not skip:
            context_x, context_y, context_sizes = self._get_context()

            self.model.set_context(context_x, context_y)
            std = self.model.get_latent_std(
                current_s[:, None],
                control=control[:, None],
                context_sizes=context_sizes[:])

        else:
            batch_size = self.history_y.size(0)
            latent_size = self.model.nn_model.z_latent_dim
            # Pred is likely very bad if there is no context
            std = torch.zeros((batch_size, latent_size), device=self.device) + torch.tensor(1, device=self.device)

        return std

    def add_interaction_idx(self, step_count):
        self.interaction_idx.append(step_count)

    def get_attentions(self, curr_y, control, step_count):
        """
        Returns the attentions of the model if it has any
        Args:
            step:

        Returns:

        """
        assert self.model is not None, "Model is not set"
        from src.modules.nd_attcnp import NdAttCNP
        assert isinstance(self.model.nn_model, NdAttCNP), "Model is not an attention model"
        if self.history_y.size(1) > 1:
            cross_attention = self.model.nn_model.encoder.cross_attention.last_attn_weights
        else:
            batch_size = self.history_y.size(0)
            cross_attention = torch.zeros((batch_size, 1, 1), device=self.device) + torch.tensor(1, device=self.device)

        return cross_attention

    def _fill_small_context(self, context_x, context_y, final_size):
        num_to_add = final_size - context_x.size(0)
        if num_to_add > 0:
            context_x = torch.concatenate([torch.zeros((num_to_add, context_x.size(1)),
                                                        device=context_x.device), context_x], dim=0)
            context_y = torch.concatenate([torch.zeros((num_to_add, context_y.size(1)),
                                                        device=context_y.device), context_y], dim=0)
        return context_x, context_y

    def _get_allowed_context_idxes(self, start_index, end_index):
        """
        Returns indices of the context that are not removed because of external interactions that
        would destroy the context
        """
        # Create an index tensor range
        idxs = torch.arange(start_index, end_index, device=self.device) + 1

        # Create a tensor from interaction indices
        interaction_indices_tensor = torch.tensor(self.interaction_idx, device=self.device)

        # Check where idxs exist in interaction_idx
        mask = torch.isin(idxs, interaction_indices_tensor, invert=True)

        # Apply mask
        allowed_indices = torch.arange(0, end_index-start_index, device=self.device)[mask]

        return allowed_indices