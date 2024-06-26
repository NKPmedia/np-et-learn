from typing import Sequence

import torch
from jaxtyping import Float
from torch import Tensor


class Interactor:
    def __init__(self):
        self.batch_size = None
        self.device = "cpu"
        self.channel_size = None

    def reset(self, batch_size: int, channel_size: int):
        self.batch_size = batch_size
        self.channel_size = channel_size

    def to(self, device):
        self.device = device

    def get_interactions(self, step: int) -> Float[Tensor, "batch state"]:
        """
        Returns the external interaction/disturbance to the system at the current step.
        Args:
            step:

        Returns:

        """
        assert self.batch_size is not None, "Interactor.reset() must be called before Interactor.get_interactions()"
        return torch.zeros((self.batch_size, self.channel_size), device=self.device)

    def does_something(self) -> bool:
        return False


class ConstantInteractor(Interactor):
    """
    Does return a fixed interaction for every batch at specific steps
    The interaction can be different at every timestep
    """

    def __init__(self,
                 interaction_steps: Sequence[int],
                 interactions: Sequence[Sequence[float]]):
        super().__init__()
        self.interaction_steps = interaction_steps
        self.interactions: Float[Tensor, "interactions state"] = torch.tensor(interactions, device=self.device)

    def to(self, device):
        super().to(device)
        self.interactions = self.interactions.to(device)

    def get_interactions(self, step: int) -> Float[Tensor, "batch channel_size"]:
        assert self.batch_size is not None, "Interactor.reset() must be called before Interactor.get_interactions()"
        if step in self.interaction_steps:
            interaction_idx = self.interaction_steps.index(step)
            return self.interactions[interaction_idx][None].repeat(self.batch_size, 1)
        else:
            return torch.zeros(self.batch_size, self.interactions.shape[1], device=self.device)

    def does_something(self) -> bool:
        if len(self.interaction_steps) > 0:
            return True
        else:
            return False