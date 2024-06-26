from typing import Tuple

import numpy as np
import torch
from jaxtyping import Integer
from torch import Tensor, Generator
from torch.distributions import Distribution


class IdxSampler:
    """
    Abstract class to sample indices for context and target points.
    """

    def __init__(self,
                 num_context_distribution: Distribution,
                 num_target_distribution: Distribution,
                 min_context_points: int = 1,
                 max_context_points: int = -1,
                 min_target_points: int = 1,
                 max_target_points: int = -1,
                 random_number_context: bool = True,
                 random_number_target: bool = True,
                 ):
        self.num_context_distribution = num_context_distribution
        self.num_target_distribution = num_target_distribution
        self.min_context_points = min_context_points
        self.max_context_points = max_context_points
        self.min_target_points = min_target_points
        self.max_target_points = max_target_points
        self.random_number_context = random_number_context
        self.random_number_target = random_number_target

    def sample_num_context_points(self, available_points: int, generator: Generator = None) -> int:
        """
        Samples the number of context points. Must be in a seperate function because the number of context points
        have to be the same over all tasks.
        If max_context_points is -1, then max_context_points = available_points
        If random_number_context is False then num_context_points = max_context_points
        Args:
            available_points: Number of available points

        Returns:
            Number of context points
        """
        max_context_points = available_points if self.max_context_points == -1 else self.max_context_points

        if self.random_number_context:
            num_context_points = self._sample_dist(self.num_context_distribution, generator)
            while (num_context_points > max_context_points) or (num_context_points < self.min_context_points):
                num_context_points = self._sample_dist(self.num_context_distribution, generator)
        else:
            num_context_points = max_context_points

        return num_context_points

    def sample(self,
               available_points: int,
               num_context_points: int,
               num_target_points: int,
               generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        """
        Samples the indices for context and target points.
        Num context points must be given because the number of context points have to be the same over all tasks.

        Args:
            num_target_points:  Number of target points
            available_points: Number of available points
            num_context_points: Number of context points
            generator: Random generator. If None the default generator is used

        Returns:
            Indices for context and target points
        """
        context_idx, target_idx = self._sample(available_points,
                                               num_context_points,
                                               num_target_points,
                                               generator=generator)

        return context_idx, target_idx

    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        raise NotImplemented()

    def _get_random_idx(self, available_points: int, num: int, generator: Generator = None) -> Integer[Tensor, "_"]:
        idx = torch.arange(0, available_points)
        idx = idx[torch.randperm(available_points, generator=generator)]
        idx_selected = idx[0: num]
        return idx_selected

    def sample_num_num_target_points_points(self, available_points, generator: Generator = None):
        """
        Samples the number of target points. Must be in a seperate function because the number of target points
        have to be the same over all tasks.
        If max_target_points is -1, then max_target_points = available_points
        If random_number_target is False then num_target_points = max_target_points
        Args:
            available_points: Number of available points

        Returns:
            Number of target points
        """
        max_target_points = available_points if self.max_target_points == -1 else self.max_target_points

        if self.random_number_target:
            num_target_points = self._sample_dist(self.num_target_distribution, generator)
            while (num_target_points > max_target_points) or (num_target_points < self.min_target_points):
                num_target_points = self._sample_dist(self.num_target_distribution, generator)
        else:
            num_target_points = max_target_points

        return num_target_points

    def _sample_dist(self, dist: Distribution, generator: Generator = None):
        if generator is not None and isinstance(dist, torch.distributions.Uniform):
            min_val = dist.low.to(int).item()
            max_val = dist.high.to(int).item()
            return torch.randint(min_val, max_val, (1,), generator=generator).to(torch.int).item()
        return dist.sample().to(torch.int).item()


class AllRandomIdxSampler(IdxSampler):
    """
    Samples all indices randomly and uniformly.
    """

    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        idx_context = self._get_random_idx(available_points, num_context_points, generator)
        idx_target = self._get_random_idx(available_points, num_target_points, generator)
        return idx_context, idx_target


class TrajectoryPartIdxSampler(IdxSampler):
    """
    Interprets the data as system trajectories and samples a random part of the trajectory.
    Use dropout to sample regular parts of the trajectory.
    """

    def __init__(self, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.dropout = dropout

    def _get_random_traj_part(self, available_points: int, num: int, dropout: float, generator: Generator = None) -> \
            Integer[Tensor, "_"]:
        start_train_idx = torch.randint(0, available_points - num + 1, (1,), generator=generator).item()
        end_train_idx = start_train_idx + num
        idx = torch.arange(start_train_idx, end_train_idx)
        idx = idx[torch.randperm(idx.size(0))]
        idx_selected = idx[0: max(int(num * (1 - dropout)), 1)]
        return idx_selected

    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        idx_context = self._get_random_traj_part(available_points, num_context_points, self.dropout, generator)
        idx_target = self._get_random_traj_part(available_points, num_target_points, self.dropout, generator)
        return idx_context, idx_target


class ConnectedTrajectoryPartIdxSampler(IdxSampler):
    """
    Interprets the data as system trajectories and samples a random part of the trajectory.
    The target data is the trajectory part after the context data.
    """

    def __init__(self, allow_shorter_context: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.allow_shorter_context = allow_shorter_context

    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        total_points = num_context_points + num_target_points
        if self.allow_shorter_context:
            minimal_target_start_idx = 0
        else:
            minimal_target_start_idx = num_context_points

        target_start_idx = torch.randint(minimal_target_start_idx, available_points - num_target_points + 1, (1,),
                                         generator=generator).item()
        target_end_idx = target_start_idx + num_target_points
        context_start_idx = max(target_start_idx - num_context_points, 0)
        context_end_idx = target_start_idx

        idx_context = torch.arange(context_start_idx, context_end_idx)
        idx_target = torch.arange(target_start_idx, target_end_idx)
        return idx_context, idx_target


class RandomContextTrajectoryPartTargetIdxSampler(TrajectoryPartIdxSampler):
    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        idx_context = self._get_random_idx(available_points, num_context_points, generator)
        idx_target = self._get_random_traj_part(available_points, num_target_points, self.dropout, generator)
        return idx_context, idx_target


class TrajectoryPartContextRandomTargetIdxSampler(TrajectoryPartIdxSampler):
    def _sample(self, available_points, num_context_points, num_target_points, generator: Generator = None) -> Tuple[
        Integer[Tensor, "_"], Integer[Tensor, "_"]]:
        idx_context = self._get_random_traj_part(available_points, num_context_points, self.dropout, generator)
        idx_target = self._get_random_idx(available_points, num_target_points, generator)
        return idx_context, idx_target
