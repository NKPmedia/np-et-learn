from logging import getLogger

import numpy as np
import torch
from torch import Tensor
from typeguard import typechecked

logger = getLogger(__name__)

class ExpRunner():
    """ experiment runner
    """

    def __init__(self):
        pass

    def run(self, env, controller, planner, device: str = "cpu"):
        """
        Runs one episode of the experiment given an environment, controller, and planner
        Args:
            env: The simulation environment and dynamics (Also defines the goal)
            controller: The controller
            planner: The planner
            device: The device to run the experiment on (cpu or cuda)
        """
        assert device in ["cpu", "cuda"]
        self.device = torch.device(device)
        controller.to(self.device)
        env.to(self.device)
        planner.to(self.device)

        done = False
        curr_y = env.reset()
        goal = env.get_goal()
        history_y, history_u, history_g = [], [], []
        step_count = 0
        score = 0.

        while not done:
            if (step_count % 50) == 0:
                print(f"Step = {step_count}")
            logger.debug("Step = {}".format(step_count))
            # plan
            g_xs = planner.plan(curr_y, goal)

            # obtain sol
            u = controller.obtain_sol(curr_y, g_xs[1:])

            # step
            next_y, cost, done = env.step(u, step_count)

            # save
            history_u.append(u)
            history_y.append(curr_y)
            history_g.append(g_xs[0])
            # update
            curr_y = next_y
            score += cost
            step_count += 1

        logger.debug("Controller type = {}, Score = {}"
                     .format(controller, score))
        return torch.stack(history_y), torch.stack(history_u), torch.stack(history_g)