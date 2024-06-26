from typing import Dict, Union, List

import torch
from jaxtyping import Float
from torch import Tensor

from src.data.parameter_changer import ParameterChangeable
from src.systems.systems.dynamic_system import ExactDynamicSystem, VanDerPolType

from src.utils.visualization.dynamic_system import visualize_2d_trajectories


class VanDerPol(ExactDynamicSystem, ParameterChangeable):

    def get_parameter_names(self) -> List[str]:
        return ["u"]

    def __init__(self, delta_t: float = 0.02):
        super().__init__(parameter_names=["u"], delta_t=delta_t, system_type=VanDerPolType())
        self.u = 1

        self.example_init_states_ranges = [[-1, 1], [-1, 1]]
        self.example_parameter_ranges = {"u": [0.1, 1.8]}

    def description(self):
        return super().description().update({
            "u": self.u
        })

    def get_parameters(self) -> Dict[str, float]:
        return {"u": self.u}

    def derivation(self,
                   state: Float[Tensor, "*batch 2"],
                   control: Float[Tensor, "*batch 1"]) -> \
            Float[Tensor, "*time 2"]:
        """
        Calculates the derivation of the ODE
        Equations from:
        https://en.wikipedia.org/wiki/Van_der_Pol_oscillator
        Args:
            state: current state, shape(*batch, 2)
            control: control input, shape(*batch, 1)

        Returns:
            derivation: derivation, shape(*time, 2)
        """
        if state.dim() == 2:
            state_per = state.permute((1, 0))
            control_per = control.permute((1, 0))
        else:
            state_per = state
            control_per = control
        x = state_per[0]
        y = state_per[1]
        c = control_per[0]

        xd = y
        yd = -x + self.u * (1 - x ** 2) * y + c

        if state.dim() == 2:
            return torch.stack([xd, yd], dim=1)
        return torch.Tensor([xd.item(), yd.item()])


if __name__ == "__main__":
    syst = VanDerPol()
    syst.u = 1
    traj = syst.predict_traj_with_init(torch.Tensor([[0.001, 0.001]]), time=30)

    visualize_2d_trajectories(traj,
                              state_names=["x", "y"],
                              trajectorie_names=["system"])
