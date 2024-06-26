import copy
import json
import os
import pickle
import pprint
from contextlib import nullcontext
from typing import Union, List, Dict, Tuple, Mapping, Sequence
import jsonpickle
import h5py
import numpy as np
import torch
from jaxtyping import Float
from lightning_fabric import seed_everything
from pytorch_lightning.utilities.seed import isolate_rng
from torch import Tensor
from tqdm import tqdm

from src.data.dataset.base import BasePckHdf5Loader
from src.data.dataset.idx_sampler import IdxSampler
from src.data.dataset.trajectory_sampler import TrajectorySampler, NormalTrajectorySampler
from src.data.parameter_changer import ParameterChanger, ConstantParameter
from src.data.utils import _up_rank
from omegaconf import DictConfig, ListConfig, OmegaConf
from src.systems.systems.dynamic_system import ExactDynamicSystem
from torch.utils.data import Dataset
from typeguard import typechecked

from src.data.noise_adders import NoiseAdder, ProcessAndObservationNoiseAdder
from src.utils.torch_utils import t2np, stack_and_pad_same_context_size


class DynamicSystemTrajectoriesGenerator(Dataset):
    """Generates trajectories of a dynamic system
    Incorporates process noise and measurement noise"""

    def __init__(self,
                 system: ExactDynamicSystem,
                 number_samples: int,
                 parameter_sample_ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]],
                 state_sample_ranges: Mapping[str, Sequence[Sequence[int]]],
                 parameter_changer: ParameterChanger = ConstantParameter(),
                 total_number_of_points: int = 100,
                 sim_time_delta: float = 0.02,
                 progress_bar: bool = False,
                 number_unrolls_per_sample: int = 500,
                 noise_adder: ProcessAndObservationNoiseAdder = None,
                 control_sample_ranges: Union[Sequence[Sequence[float]]] = []):
        """Initialization"""
        self.generation_arguments_snapshot = locals()
        self.generation_arguments_snapshot["self"] = None
        self.parameter_changer = parameter_changer
        self.parameter_changer.set_init_sample_ranges(parameter_sample_ranges)
        self.number_samples = number_samples
        self.parameter_sample_ranges = parameter_sample_ranges
        self.state_sample_ranges = state_sample_ranges
        self.dynamic_system = system
        self.total_number_of_points = total_number_of_points
        self.sim_time_delta = sim_time_delta
        self.progress_bar = progress_bar
        self.number_unrolls_per_sample = number_unrolls_per_sample
        self.control_sample_ranges = Tensor(control_sample_ranges)
        self.noise_adder = noise_adder

        self.sampler = NormalTrajectorySampler()
        self.noise_adder.state_idx = list(range(system.state_size))

    def description(self):
        return {
            "system": self.dynamic_system.description(),
            "parameter_sample_ranges": OmegaConf.to_object(self.parameter_sample_ranges),
            "state_sample_ranges": OmegaConf.to_object(self.state_sample_ranges),
            "total_number_of_points": self.total_number_of_points,
            "number_unrolls_per_sample": self.number_unrolls_per_sample,
        }

    def __getitem__(self, index):
        """Generates one sample of data"""
        task = {'x': []}

        if self.progress_bar:
            prog = tqdm(index)
        else:
            prog = index
        for id in prog:
            self.parameter_changer.reset(self.dynamic_system)

            x, y = self.sampler.sample_trajectory(self.dynamic_system,
                                                  self.total_number_of_points,
                                                  self.number_unrolls_per_sample,
                                                  self.state_sample_ranges,
                                                  self.control_sample_ranges,
                                                  self.noise_adder,
                                                  self.parameter_changer)

            for i in range(x.shape[0]):
                task['x'].append(x[i])

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_up_rank(np.stack(v, axis=0)),
                                dtype=torch.float32)
                for k, v in task.items()}
        return task

    def save(self, path: str):
        """
        Saved all the data in a hdf5 file
        Saves a json version of this object
        Args:
            path:

        Returns:

        """
        data = self[range(self.number_samples)]
        data = {key: value.numpy() for key, value in data.items()}

        f = h5py.File(path, 'w-')
        for key, value in data.items():
            dset = f.create_dataset(key, value.shape, dtype=np.float32, fletcher32=True)
            dset[...] = t2np(value)
        f.close()
        param_path = os.path.splitext(path)[0] + ".json"
        frozen = jsonpickle.encode(self.generation_arguments_snapshot, indent=4, separators=(',', ': '))
        with open(param_path, 'w') as f:
            f.write(frozen)


class DynamicSystemInitStateGenerator(Dataset):
    """Generates trajectories of a dynamic system
    Incorporates process noise and measurement noise"""

    def __init__(self,
                 system: ExactDynamicSystem,
                 number_samples: int,
                 parameter_sample_ranges: Union[Mapping[str, Sequence[float]], Mapping[str, float]],
                 state_sample_ranges: Sequence[Sequence[int]],
                 parameter_changer: ParameterChanger = ConstantParameter(),
                 total_number_of_points: int = 100,
                 sim_time_delta: float = 0.02,
                 progress_bar: bool = False,
                 number_unrolls_per_sample: int = 500):
        """Initialization"""
        self.generation_arguments_snapshot = locals()
        self.generation_arguments_snapshot["self"] = None
        self.parameter_changer = parameter_changer
        self.parameter_changer.set_init_sample_ranges(parameter_sample_ranges)
        self.number_samples = number_samples
        self.parameter_sample_ranges = parameter_sample_ranges
        self.state_sample_ranges = state_sample_ranges
        self.dynamic_system = system
        self.total_number_of_points = total_number_of_points
        self.sim_time_delta = sim_time_delta
        self.progress_bar = progress_bar
        self.number_unrolls_per_sample = number_unrolls_per_sample

    def description(self):
        return {
            "system": self.dynamic_system.description(),
            "parameter_sample_ranges": OmegaConf.to_object(self.parameter_sample_ranges),
            "state_sample_ranges": OmegaConf.to_object(self.state_sample_ranges),
            "total_number_of_points": self.total_number_of_points,
            "number_unrolls_per_sample": self.number_unrolls_per_sample,
        }

    def __getitem__(self, index):
        """Generates one sample of data"""
        task = {'x': [],
                "parameter": []}

        if self.progress_bar:
            prog = tqdm(index)
        else:
            prog = index
        for id in prog:
            self.parameter_changer.reset(self.dynamic_system)
            parameters = []
            init_state = self.dynamic_system.sample_init_state(self.state_sample_ranges,
                                                               number=self.number_unrolls_per_sample)
            for i in range(self.total_number_of_points):
                self.parameter_changer.parameter_step(self.dynamic_system, i)
                parameter = self.dynamic_system.get_parameters_as_list()
                parameters.append(torch.concatenate(parameter))

            for i in range(init_state.size(0)):
                task['x'].append(init_state[i])
            for i in range(init_state.size(0)):
                task['parameter'].append(torch.stack(parameters, dim=0))

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(torch.stack(v, dim=0),
                                dtype=torch.float32)
                for k, v in task.items()}
        return task

    def save(self, path: str):
        """
        Saved all the data in a hdf5 file
        Saves a json version of this object
        Args:
            path:

        Returns:

        """
        data = self[range(self.number_samples)]
        data = {key: value.numpy() for key, value in data.items()}

        f = h5py.File(path, 'w-')
        for key, value in data.items():
            dset = f.create_dataset(key, value.shape, dtype=np.float32, fletcher32=True)
            dset[...] = t2np(value)
        f.close()
        param_path = os.path.splitext(path)[0] + ".json"
        frozen = jsonpickle.encode(self.generation_arguments_snapshot, indent=4, separators=(',', ': '))
        with open(param_path, 'w') as f:
            f.write(frozen)


class DynamicSystemDatasetGenerator(Dataset):

    def __init__(self,
                 system: ExactDynamicSystem,
                 number_samples: int,
                 context_trajectory_sampler: TrajectorySampler,
                 target_trajectory_sampler: TrajectorySampler,
                 parameter_sample_ranges: Union[Dict["str", List[float]], DictConfig["str", ListConfig[float]],
                 Dict["str", float], DictConfig["str", float]],
                 state_sample_ranges: DictConfig["str", ListConfig[ListConfig[int]]],
                 total_number_of_points: int = 100,
                 sim_time_delta: float = 0.02,
                 progress_bar: bool = False,
                 number_unrolls_per_sample: int = 500,
                 control_sample_ranges: Union[List[List[float]], ListConfig[ListConfig[int]]] = []):
        """Initialization"""
        self.generation_arguments_snapshot = locals()
        self.generation_arguments_snapshot["self"] = None
        self.number_samples = number_samples
        self.parameter_sample_ranges = parameter_sample_ranges
        self.state_sample_ranges = state_sample_ranges
        self.dynamic_system = system
        self.total_number_of_points = total_number_of_points
        self.sim_time_delta = sim_time_delta
        self.progress_bar = progress_bar
        self.number_unrolls_per_sample = number_unrolls_per_sample
        self.control_sample_ranges = Tensor(control_sample_ranges)
        self.target_trajectory_sampler = target_trajectory_sampler
        self.context_trajectory_sampler = context_trajectory_sampler

    def description(self):
        return {
            "system": self.dynamic_system.description(),
            "parameter_sample_ranges": OmegaConf.to_object(self.parameter_sample_ranges),
            "state_sample_ranges": OmegaConf.to_object(self.state_sample_ranges),
            "total_number_of_points": self.total_number_of_points,
            "target_trajectory_sampler": self.target_trajectory_sampler.__class__.__name__,
            "context_trajectory_sampler": self.context_trajectory_sampler.__class__.__name__,
            "number_unrolls_per_sample": self.number_unrolls_per_sample,
        }

    def __getitem__(self, index):
        """Generates one sample of data"""
        task = {'x': [],
                'y': [],
                'x_context': [],
                'y_context': []}

        if self.progress_bar:
            prog = tqdm(index)
        else:
            prog = index
        for id in prog:
            parameter = self.dynamic_system.sample_system_parameter(self.parameter_sample_ranges)
            self.dynamic_system.set_parameter(parameter)

            x_context, y_context = self.context_trajectory_sampler.sample_trajectory(self.dynamic_system,
                                                                                     self.total_number_of_points,
                                                                                     self.number_unrolls_per_sample,
                                                                                     self.state_sample_ranges,
                                                                                     self.control_sample_ranges)

            x, y = self.target_trajectory_sampler.sample_trajectory(self.dynamic_system,
                                                                    self.total_number_of_points,
                                                                    self.number_unrolls_per_sample,
                                                                    self.state_sample_ranges,
                                                                    self.control_sample_ranges)

            self._append_data(task, x, y, x_context, y_context)

        # Stack batch and convert to PyTorch.
        task = {k: torch.tensor(_up_rank(np.stack(v, axis=0)),
                                dtype=torch.float32)
                for k, v in task.items()}
        return task

    def _append_data(self, task,
                     x: Float[torch.Tensor, "batch time in"],
                     y: Float[torch.Tensor, "batch time out"],
                     x_context: Float[torch.Tensor, "batch time in"],
                     y_context: Float[torch.Tensor, "batch time out"]):
        for i in range(x.shape[0]):
            task['x'].append(x[i])
            task['y'].append(y[i])
            task['x_context'].append(x_context[i])
            task['y_context'].append(y_context[i])


class OfflineDynamicSystemDataset(BasePckHdf5Loader):

    def __init__(self,
                 path: str,
                 noise_adder: NoiseAdder,
                 idx_sampler: IdxSampler,
                 preload: bool = True,
                 include_context_in_target: bool = False,
                 different_context_sizes_in_batch: bool = False,
                 val_set: bool = False,
                 context_mode: str = "separate"):
        """Initialization"""
        super().__init__(path, preload)
        self.noise_adder = noise_adder
        self.index_sampler = idx_sampler
        self.include_context_in_target = include_context_in_target
        self.different_context_sizes_in_batch = different_context_sizes_in_batch
        self.val_set = val_set
        self.context_mode = context_mode

    def __len__(self):
        """Denotes the total number of samples"""
        return self.db["x"].shape[0]

    def __getitem__(self, index):
        with isolate_rng(include_cuda=False) if self.val_set else nullcontext():
            if self.val_set:
                torch.manual_seed(sum(index))

            """Generates one sample of data"""
            fixed_size_task_part = \
                {'x': [],
                 'y': [],
                 'x_target': [],
                 'y_target': [],
                 'target_context_idx': [],
                 'target_target_idx': []}
            variable_size_task_part = \
                {'x_context': [],
                 'y_context': [],
                 }
            assert self.db["x"].shape[1] == self.db["x_context"].shape[1]

            noise_adder = copy.deepcopy(self.noise_adder)

            available_points = self.db["x"].shape[1]

            if not self.different_context_sizes_in_batch:
                num_context_points = self.index_sampler.sample_num_context_points(available_points)
            num_target_points = self.index_sampler.sample_num_num_target_points_points(available_points)

            context_sizes = []
            target_sizes = []
            for id in index:
                if self.different_context_sizes_in_batch:
                    num_context_points = self.index_sampler.sample_num_context_points(available_points)


                if self.context_mode == "separate":
                    x = self.db["x"][id].clone().detach()
                    y = self.db["y"][id].clone().detach()
                    x_context = self.db["x_context"][id].clone().detach()
                    y_context = self.db["y_context"][id].clone().detach()
                else:
                    # randomly choose x, y from x,y or x_context, y_context
                    choice = torch.randint(0, 2, [1])
                    if choice == 0:
                        x = self.db["x"][id].clone().detach()
                        y = self.db["y"][id].clone().detach()
                    else:
                        x = self.db["x_context"][id].clone().detach()
                        y = self.db["y_context"][id].clone().detach()

                context_idx, target_idx = self.index_sampler.sample(available_points, num_context_points, num_target_points)
                context_sizes.append(len(context_idx))

                noise_adder.sample_noise_lvl()
                x_context, y_context, x_target, y_target = noise_adder.add_noise_to_data(
                    x_context[context_idx], y_context[context_idx], x[target_idx], y[target_idx]
                )

                if self.include_context_in_target:
                    x_target = torch.cat([x_context, x_target], dim=0)
                    y_target = torch.cat([y_context, y_target], dim=0)
                    target_context_idx = torch.arange(0, len(context_idx)).to(torch.int)
                    target_target_idx = torch.arange(len(context_idx), len(context_idx) + len(target_idx)).to(torch.int)
                    target_sizes.append(len(context_idx) + len(target_idx))
                else:
                    target_context_idx = torch.arange(0, 0).to(torch.int)
                    target_target_idx = torch.arange(0, len(target_idx)).to(torch.int)
                    target_sizes.append(len(target_idx))

                # Record to task.
                fixed_size_task_part['x'].append(x)
                fixed_size_task_part['y'].append(y)
                fixed_size_task_part['target_context_idx'].append(target_context_idx)
                fixed_size_task_part['target_target_idx'].append(target_target_idx)
                variable_size_task_part['x_context'].append(x_context)
                variable_size_task_part['y_context'].append(y_context)
                fixed_size_task_part['x_target'].append(x_target)
                fixed_size_task_part['y_target'].append(y_target)

            # Stack batch and convert to PyTorch.
            fixed_size_task_part = {k: _up_rank(torch.stack(v, dim=0))
                    for k, v in fixed_size_task_part.items()}
            variable_size_task_part = {k: _up_rank(stack_and_pad_same_context_size(v))
                                    for k, v in variable_size_task_part.items()}

            task = {**fixed_size_task_part,
                    **variable_size_task_part,
                    'context_sizes': context_sizes,
                    'target_sizes': target_sizes}
        return task
