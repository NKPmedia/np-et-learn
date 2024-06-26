from typing import List, Iterable, Sequence

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

from src.utils.torch_utils import t2np


def visualize_2d_trajectories(states: Float[torch.Tensor, "batch time 2"],
                              trajectorie_names: List[str] = [],
                              state_names: List[str] = [],
                              std: Float[torch.Tensor, "batch time 2"] = None,
                              nn_idx: int = 1,
                              title: str = None,
                              ):
    fig = plt.figure(figsize=(8, 8))
    axs = fig.add_subplot(1, 1, 1)
    for i in range(states.shape[0]):
        if state_names:
            axs.set_xlabel(state_names[0])
            axs.set_ylabel(state_names[1])
        axs.plot(t2np(states[i, :, 0].T), t2np(states[i, :, 1].T), label=trajectorie_names[i])
    if std is not None:
        for t in range(states.shape[1]):
            axs.add_artist(Ellipse((states[nn_idx][t][0], states[nn_idx][t][1]),
                                   std[0, t, 0], std[0, t, 1], color='g', alpha=0.4))
    fig.legend()
    if title is not None:
        fig.suptitle(title)
    fig.show()


def visualize_nd_trajectories(trajectories: Float[torch.Tensor, "batch time channel"],
                              trajectory_names: Sequence[str] = (), y_bounds: Sequence[float] = (),
                              channel_names: Sequence[str] = (), delta_t: float = 0.02,
                              std: Float[torch.Tensor, "batch time channel"] = None, std_trajectory_idx: Sequence[int] = (1,),
                              title: str = None, number_plot_cols: int = 2):
    """
    Visualizes a number of trajectories in a 2xi plot
    Adds more rows if there are more than 2 channels
    The order of the variance channels must match the order of the trajectories in variance_trajectory_idx
    Args:
        trajectories: shape: (batch, time, channel)
        trajectory_names: List of names for the trajectories
        y_bounds: Tuple of lower and upper bound for the y axis
        channel_names: List of names for the channels
        delta_t: time step to calculate the time axis based on the number of steps
        std: Standard deviation for some of the trajectories; shape: (batch, time, channel)
        std_trajectory_idx: List of indices for which the standart deviation is given
        title: Title of the plot
        number_plot_cols: Number of columns in the plot

    """
    assert len(channel_names) == trajectories.shape[-1], \
        f"Numer of state {len(channel_names)} names must match the number of states {trajectories.shape[-1]}"
    if trajectory_names:
        assert len(trajectory_names) == trajectories.shape[0], \
            f"Number of trajectory names {len(trajectory_names)} must match the number of trajectories {trajectories.shape[0]}"

    rows = int(np.ceil(trajectories.shape[-1] / number_plot_cols))

    fig = plt.figure(figsize=(6*number_plot_cols, rows * 3))
    steps = trajectories.shape[1]
    time = np.arange(0, steps)[None] * delta_t
    for i in range(trajectories.shape[2]):
        axs = fig.add_subplot(rows, number_plot_cols, i + 1)
        if channel_names:
            name = channel_names[i]
        else:
            name = f"Trac {i}"
        axs.set_xlabel("t [s]")
        axs.set_ylabel(name)
        if y_bounds:
            axs.set_ylim(y_bounds[0], y_bounds[1])
        if i == 0 and trajectory_names:
            axs.plot(np.repeat(time, trajectories.shape[0], axis=0).T, t2np(trajectories[:, :, i]).T, label=trajectory_names)
        else:
            axs.plot(np.repeat(time, trajectories.shape[0], axis=0).T, t2np(trajectories[:, :, i]).T)
        if std is not None:
            for j, std_idx in enumerate(std_trajectory_idx):
                axs.fill_between(time[0],
                                 trajectories[std_idx, :, i] - 2 * std[j, :, i],
                                 trajectories[std_idx, :, i] + 2 * std[j, :, i],
                                 color="g",
                                 alpha=0.2)
    if title is not None:
        fig.suptitle(title)
    fig.legend()
    fig.show()
