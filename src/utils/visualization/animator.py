import os
from logging import getLogger
from typing import Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import torch
from jaxtyping import Float
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from torch import Tensor

from src.systems.runner.predict_test_runner import EnvRunResult

logger = getLogger(__name__)


class Animator():
    """ animation class
    """

    def __init__(self, results: Sequence[EnvRunResult],
                 plot_func, name: str = "env",
                 controller_type: str = "controller",
                 labels: Union[Sequence[str], None] = None):
        """
        """
        self.env_name = name
        self.controller_type = controller_type

        self.interval = results[0].delta_t * 1000.  # to ms
        self.plot_func = plot_func
        self.results = results
        self.imgs = []

        self.labels = labels

    def _setup(self, batch_size: int):
        """ set up figure of animation
        """
        # make fig
        self.anim_fig = plt.figure()

        # axis
        self.axis = self.anim_fig.add_subplot(111)
        self.axis.set_aspect('equal', adjustable='box')

        self.imgs = self.plot_func(self.axis, batch_size=batch_size, labels=self.labels)

        if self.labels is not None:
            self.anim_fig.legend(loc='upper right')

    def _update_img(self, i, history_x: Float[Tensor, "batch time state"]):
        """ update animation

        Args:
            i (int): frame count
            history_x: history of states shape: (batch, time, state)
        """
        self.plot_func(self.imgs, i, history_s=history_x)

    def draw(self, save_path: str):
        """draw the animation and save
        Returns:
            None
        """
        # set up animation figures

        history_x = torch.stack([res.system.system_states for res in self.results])

        self._setup(history_x.size(0))
        def _update_img(i): return self._update_img(i, history_x)

        # Set up formatting for the movie files
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=50, metadata=dict(artist='Me'), bitrate=1800)

        # call funcanimation
        ani = FuncAnimation(
            self.anim_fig,
            _update_img, interval=self.interval, frames=history_x.size(1)-1)

        # save animation
        path = os.path.join(save_path, "animation.mp4")
        logger.info("Saved Animation to {} ...".format(path))

        os.makedirs(save_path, exist_ok=True)
        ani.save(path, writer=writer)

