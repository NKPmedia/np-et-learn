from types import NoneType
from typing import List, Tuple, Union, Dict

import torch
from jaxtyping import Float, Bool
from torch import Tensor

from src.systems.predictor.moving_window_nn_predictor import MovingWindowNNPredictor
from src.systems.predictor.predictor import Predictor
from src.systems.predictor.reset.reset_trigger import ResetTrigger, ConstantTrigger
from src.systems.systems.NNModel import NNModel


class FixedWindowNNPredictor(MovingWindowNNPredictor):
    """
    Selects the context points from a set of trajectories using a fixed window at the beginning of the trajectory
    """

    def __init__(self, model: NNModel = None, window_size: Dict = {0: 200},
                 reset_trigger: ResetTrigger = ConstantTrigger([]),
                 fill_small_windows: bool = False,
                 log_zeroControl_unroll: bool = False, unroll_mode: str = "mean_propagation"):
        """
        Args:
            window_size: size of the moving window
        """
        super().__init__(model=model, reset_trigger=reset_trigger,
                     fill_small_windows=fill_small_windows, unroll_mode=unroll_mode,
                         log_zeroControl_unroll=log_zeroControl_unroll)
        self.window_size = window_size

    def _update_window_starts(self, trigger: Bool[Tensor, "batch"], history_len: int, window_size: int):
        """
        Sets the window starts to 0
        The fixed window always starts at the beginning of the trajectory
        Args:
            trigger:
            history_len:
            window_size:

        Returns:

        """
        step = history_len - 1
        if self.window_starts is None:
            self.window_starts = torch.zeros_like(trigger, dtype=torch.long)
        self.window_starts[trigger] = step - 1

        window_size = self.get_window_size_by_start(self.window_starts)

        current_window_size = torch.max(torch.zeros_like(step - window_size), step - window_size)
        current_window_size[current_window_size > window_size] = window_size[current_window_size > window_size]

        if self.actual_window_size is None:
            self.actual_window_size = current_window_size[:, None]
        else:
            self.actual_window_size = torch.concatenate([self.actual_window_size, current_window_size[:, None]], dim=1)

    def _get_context(self, batch_nr: int = None) -> \
            Tuple[Float[Tensor, "batch context_size state_control"],
            Float[Tensor, "batch context_size state"],
            Float[Tensor, "batch"]]:
        """
        Selects the context points from trajectories
        The window is always at the beginning of the trajectory with a maximum size of window_size

        Args:
            batch_nr: Batch number of the context to select; if None then all the batches are selected

        Returns:
            context_x: set of states; shape(batch context_size, state_size+control_size)
            context_y: set of next_states; shape(batch context_size, state_size+control_size)
            context_sizes: list of context sizes; shape(batch)
        """
        history = torch.concatenate([self.history_y, self.history_u], dim=-1)
        if self.history_y.size(1) < 2:
            raise ValueError("Trajectory is too short for moving window context selector")

        contexts_x = []
        contexts_y = []
        context_sizes = []

        barch_range = range(history.size(0)) if batch_nr is None else [batch_nr]
        for batch in barch_range:
            start_index = self.window_starts[batch]
            window_size = self.get_window_size_by_start(start_index)
            end_index = min(start_index + window_size, history.size(1) - 1)
            context_x = history[batch, start_index:end_index]
            context_y = self.history_y[batch, start_index + 1:end_index + 1]
            contexts_x.append(context_x)
            contexts_y.append(context_y)
            context_sizes.append(context_x.size(0))
            if context_sizes[-1] == 0:
                raise ValueError("Window size is 0. Not good for NPs Error")

        contexts_x, contexts_y = self._pad_contexts(contexts_x, contexts_y)

        return contexts_x, contexts_y, torch.tensor(context_sizes)

    def get_window_size_by_start(self, start: Union[int, Tensor]):
        if isinstance(start, int):
            for key, value in self.window_size.items():
                if start >= key:
                    return value
        elif isinstance(start, Tensor):
            window_size = torch.zeros_like(start)
            for key, value in self.window_size.items():
                window_size[start >= key] = value
            return window_size
        else:
            raise ValueError("start must be int or Tensor")