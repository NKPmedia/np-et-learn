from typing import Dict, List, Union

import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from src.data.parameter_changer import ParameterChangeable
from src.systems.systems.dynamic_system import ExactDynamicSystem, InvertedPendulumType
from typeguard import typechecked


@typechecked
class InvertedPendulum(ExactDynamicSystem, ParameterChangeable):
    def get_parameter_names(self) -> List[str]:
        return ["m", "M", "L", "pole_friction", "cart_friction"]

    # Pendulum and Cart system.
    # Y : [x, x_dot, theta, theta_dot]
    # returns expression for Y_dot.
    def __init__(self, delta_t: float = 0.02):
        super().__init__(parameter_names=["m", "M", "L", "pole_friction", "cart_friction"],
                         delta_t=delta_t,
                         system_type=InvertedPendulumType())
        self.g = 9.81  # Gravitational Acceleration
        self.L = 0.20  # Length of pendulum (From dsme pendulum)

        self.m = 0.160  # mass of bob (kg) (From dsme pendulum)
        self.M = 1  # mass of cart (kg) (From Deep Pilco paper)
        self.pole_friction = 0.001 # Ns/m (From https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)
        self.cart_friction = 0.01 # Ns/m (From https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html)

        #Simmilar values like in https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html

        self.example_init_states_ranges = [[-5, 5], [-1, 1], [-6, 6], [-20, 20]]
        self.example_parameter_ranges = {"L": [0.05, 0.5]}

    def description(self):
        return {
            "name": self.__class__.__name__,
            "m": self.m,
            "M": self.M,
            "L": self.L,
            "pole_friction": self.pole_friction
        }

    def get_parameters(self) -> Dict[str, Union[float, Tensor]]:
        return {
            "m": self.m,
            "M": self.M,
            "L": self.L,
            "pole_friction": self.pole_friction,
            "cart_friction": self.cart_friction
        }

    def derivation(self, state: Float[torch.Tensor, "*time 4"], control: Float[torch.Tensor, "*time 1"]) -> \
            Float[torch.Tensor, "*time 4"]:
        """
        Equations from:
        Florian, Razvan V.
        "Correct equations for the dynamics of the cart-pole system."
        Center for Cognitive and Neural Studies (Coneural), Romania (2007).
        """
        if state.dim() == 2:
            state_per = state.permute((1, 0))
            control_per = control.permute((1, 0))
        else:
            state_per = state
            control_per = control
        x = state_per[0]
        xd = state_per[1]
        theta = state_per[2]
        thetad = state_per[3]

        l = self.L / 2 #Int he formula the pole has the length 2L

        # Nc = 1  # Assumption: The cart does not leaf the track therefore Nc is always positive
        #
        # inner_fraction1 = (-control_per[0] - self.m * l * thetad ** 2 * (
        #     torch.sin(theta) + self.cart_friction * torch.sign(xd) * torch.cos(theta))) \
        #                   / (self.m + self.M)
        # tmp1 = inner_fraction1 + self.cart_friction * self.g * torch.sign(xd)
        # theta_dott_numerator = self.g * torch.sin(theta) + torch.cos(theta) * tmp1 - (
        #             (self.pole_friction * thetad) / (self.m * l))
        #
        # inner_fraction2 = (self.m * torch.cos(theta)) / (self.m + self.M)
        # theta_dott_denominator = l * (
        #             4 / 3 - inner_fraction2 * (torch.cos(theta) - self.cart_friction * torch.sign(xd)))
        # theta_ddot = theta_dott_numerator / theta_dott_denominator
        #
        # # Calculate correct Nc with theta_ddot
        # Nc = (self.m * self.M) * self.g - self.m * l * (
        #             theta_ddot * torch.sin(theta) + thetad ** 2 * torch.cos(theta))
        #
        # assert (Nc > 0).all(), "Nc is not positive, but it was assumed to be positive. (Does the cart leave the track?)"
        #
        # x_ddot_numerator = control_per[0] + self.m * l * (
        #             thetad ** 2 * torch.sin(theta) - theta_ddot * torch.cos(theta)) \
        #                    - self.cart_friction * Nc * torch.sign(xd)
        # x_ddot_denominator = self.M + self.m
        # x_ddot = x_ddot_numerator / x_ddot_denominator


        #https://sharpneat.sourceforge.io/research/cart-pole/cart-pole-equations.html
        tmp = control_per[0] + self.m * l * thetad ** 2 * torch.sin(theta) - self.cart_friction * xd
        x_ddot_numerator = self.m*self.g*torch.sin(theta)*torch.cos(theta) \
                           - 7/3*tmp \
                           - (self.pole_friction*thetad*torch.cos(theta))/(l)
        x_ddot_denominator = self.m * torch.cos(theta)**2 - 7/3*self.M
        x_ddot = x_ddot_numerator / x_ddot_denominator

        theta_ddot = 3/(7*l) * (self.g*torch.sin(theta) - x_ddot*torch.cos(theta) - self.pole_friction*thetad/(self.m*l))

        # theta_dott_numerator = control * torch.cos(state[2]) - (self.M + self.m) * self.g * torch.sin(state[2]) + \
        #                       self.m * self.L * (torch.cos(state[2] * torch.sin(state[2]))) * (state[3] ** 2)
        # theta_dott_denominator = self.m * self.L * torch.cos(state[2]) ** 2 - (self.M + self.m) * self.L
        # theta_ddot = theta_dott_numerator / theta_dott_denominator
        #
        # x_ddot_numerator = control + self.m * self.L * torch.sin(state[2]) * (
        #        state[3] ** 2) - self.m * self.g * torch.cos(state[2]) * torch.sin(state[2])
        # x_ddot_denominator = self.M + self.m - self.m * (torch.cos(state[2]) ** 2)
        # x_ddot = x_ddot_numerator / x_ddot_denominator
        # damping_theta = - 0.5 * state[3]
        # damping_x = - 1.0 * state[1]

        if state.dim() == 2:
            return torch.stack([state_per[1], x_ddot, state_per[3], theta_ddot], dim=1)
        return torch.Tensor([state_per[1], x_ddot.item(), state_per[3], theta_ddot.item()])