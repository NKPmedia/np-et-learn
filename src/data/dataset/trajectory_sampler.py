from typing import Tuple, List, Union

import numpy as np
import torch
from jaxtyping import Float
from tqdm import tqdm

from src.data.noise_adders import ZeroNoiseAdder, ProcessAndObservationNoiseAdder
from src.data.parameter_changer import ParameterChanger, ConstantParameter
from src.systems.systems.dynamic_system import ExactDynamicSystem
from src.systems.control.controller import Controller
from src.systems.planner.planner import Planner


class TrajectorySampler():

    def __init__(self):
        self.state_sample_type = "trajectory"

    def sample_trajectory(self,
                          dynamic_system: Union[ExactDynamicSystem],
                          unroll_length: int,
                          num_unrolls: int,
                          state_sample_ranges,
                          control_sample_ranges=None,
                          noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                          parameter_changer: ParameterChanger = ConstantParameter()
                          ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        x, y = self._sample_trajectory(dynamic_system,
                                       unroll_length,
                                       num_unrolls,
                                       state_sample_ranges,
                                       control_sample_ranges,
                                       noise_adder,
                                       parameter_changer)
        if (torch.isnan(x).any()
                or torch.isnan(y).any()
                or torch.isnan(x).any()
                or torch.isnan(y).any()):
            raise Exception("Dynamic System datageneration generated nan Data!!!! Take a look at you parameters.")
        return x, y

    def _sample_trajectory(self,
                           dynamic_system: Union[ExactDynamicSystem],
                           unroll_length: int,
                           num_unrolls: int,
                           state_sample_ranges,
                           control_sample_ranges=None,
                           noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                           parameter_changer: ParameterChanger = ConstantParameter()
                           ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        raise NotImplemented()


class RandomTransitionSampler(TrajectorySampler):

    def __init__(self):
        super().__init__()
        self.state_sample_type = "random"

    def _sample_trajectory(self,
                           dynamic_system: Union[ExactDynamicSystem],
                           unroll_length: int,
                           num_unrolls: int,
                           state_sample_ranges,
                           control_sample_ranges=None,
                           noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                           parameter_changer: ParameterChanger = ConstantParameter()
                           ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        assert isinstance(parameter_changer, ConstantParameter), \
            "RandomTransitionSampler only works with ConstantParameter, because no trajectory is sampled"
        total_samples = num_unrolls * unroll_length
        if control_sample_ranges is not None and control_sample_ranges.size(0) != 0:
            control_noise = torch.rand((total_samples, dynamic_system.control_size))
            control = control_noise * (
                    control_sample_ranges[:, 1] - control_sample_ranges[:, 0]) + control_sample_ranges[:, 0]
        else:
            control = torch.zeros((total_samples, dynamic_system.control_size))

        if isinstance(dynamic_system, ExactDynamicSystem):
            x, y = self._sample_random_dynamic_system(dynamic_system,
                                                      unroll_length,
                                                      num_unrolls,
                                                      state_sample_ranges,
                                                      control,
                                                      noise_adder
                                                      )
        else:
            raise Exception("dynamic_system is not a known type.")
        return x, y

    def _sample_random_dynamic_system(self, dynamic_system, unroll_length, num_unrolls, state_sample_ranges, control,
                                      noise_adder):
        total_samples = num_unrolls * unroll_length
        x = dynamic_system.sample_init_state(state_sample_ranges[self.state_sample_type], total_samples)
        y = dynamic_system.next_state_rk4(x, control)

        x = torch.concatenate([x, control], dim=-1)
        x = x.reshape((num_unrolls, unroll_length, -1))
        y = y.reshape((num_unrolls, unroll_length, -1))
        x, y = noise_adder.add_observation_process_noise(x, y)
        return x, y


class NormalTrajectorySampler(TrajectorySampler):

    def __init__(self, precomputed_controls_path: str = None):
        super().__init__()
        self.state_sample_type = "trajectory"
        if precomputed_controls_path is not None:
            self.controls = torch.load(precomputed_controls_path)
            if self.controls.ndim == 2:
                # Add control channel dimension
                self.controls = self.controls[..., None]
        else:
            self.controls = None

    def _sample_trajectory(self,
                           dynamic_system: Union[ExactDynamicSystem],
                           unroll_length: int,
                           num_unrolls: int,
                           state_sample_ranges,
                           control_sample_ranges=None,
                           noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                           parameter_changer: ParameterChanger = ConstantParameter()
                           ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        if self.controls is None:
            if control_sample_ranges is not None and control_sample_ranges.size(0) != 0:
                control_noise = torch.rand((num_unrolls, unroll_length + 1, dynamic_system.control_size))
                control = control_noise * (
                        control_sample_ranges[:, 1] - control_sample_ranges[:, 0]) + control_sample_ranges[:, 0]
            else:
                control = torch.zeros((num_unrolls, unroll_length + 1, dynamic_system.control_size))
        else:
            assert self.controls.shape[-1] == dynamic_system.control_size, \
                "The control size of the dynamic system and the control size of the precomputed controls do not match."
            random_indexs = torch.randint(0, self.controls.shape[0], (num_unrolls,))
            control = self.controls[random_indexs, :unroll_length+1, :]
        if isinstance(dynamic_system, ExactDynamicSystem):
            x, y = self._sample_trajectory_dynamic_system(dynamic_system,
                                                          unroll_length,
                                                          num_unrolls,
                                                          state_sample_ranges,
                                                          control,
                                                          noise_adder,
                                                          parameter_changer)
        else:
            raise Exception("dynamic_system is not a known type.")
        return x, y

    def _sample_trajectory_dynamic_system(self,
                                          dynamic_system: Union[ExactDynamicSystem],
                                          unroll_length: int,
                                          num_unrolls: int,
                                          state_sample_ranges,
                                          control: Float[torch.Tensor, "batch time control"],
                                          noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                                          parameter_changer: ParameterChanger = ConstantParameter(),
                                          ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:

        x = dynamic_system.sample_init_state(state_sample_ranges[self.state_sample_type], num_unrolls)
        parameter_changer.parameter_step(dynamic_system, 0)
        y = dynamic_system.next_state_rk4(x, control[:, 0])
        y_pNoised = noise_adder.add_process_noise(x, y)
        x_all = []
        y_all = []
        for t in range(unroll_length):
            parameter_changer.parameter_step(dynamic_system, t + 1)
            x_all.append(noise_adder.add_observation_noise(x[None]))
            y_all.append(noise_adder.add_observation_noise(y_pNoised[None]))
            x = y_pNoised
            y = dynamic_system.next_state_rk4(x, control[:, t + 1])
            y_pNoised = noise_adder.add_process_noise(x, y)
        x = torch.concatenate(x_all, dim=0).permute(1, 0, 2)
        x = torch.concatenate([x, control[:, :-1, :]], dim=-1)
        y = torch.concatenate(y_all, dim=0).permute(1, 0, 2)
        return x, y

    def _sample_trajectory_env(self,
                               dynamic_system: Union[ExactDynamicSystem],
                               unroll_length: int,
                               num_unrolls: int,
                               state_sample_ranges,
                               control: Float[torch.Tensor, "batch time control"],
                               noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                               parameter_changer: ParameterChanger = ConstantParameter()
                               ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:

        assert noise_adder.process_context_noise_level == 0 and noise_adder.process_target_noise_level == 0, \
            "Envs do not support process noise."

        dynamic_system.reset(init_state=torch.zeros((num_unrolls, dynamic_system.state_size)),
                             parameter_changer=parameter_changer)
        parameter_changer.parameter_step(dynamic_system, 0)

        y_all = []
        for t in range(unroll_length + 1):
            parameter_changer.parameter_step(dynamic_system, t)
            x = dynamic_system.step(control[:, t], t)[0]
            y_all.append(noise_adder.add_observation_noise(x[None]))

        y_t = torch.concatenate(y_all, dim=0).permute(1, 0, 2)
        y_tp1 = y_t[:, 1:]
        y_t = y_t[:, :-1]
        y_t = torch.concatenate([y_t, control[:, 1:, :]], dim=-1)
        return y_t, y_tp1


class ControlledTrajectorySampler(TrajectorySampler):

    def __init__(self, controller: Controller, planer: Planner):
        super().__init__()
        self.state_sample_type = "trajectory"
        self.controller = controller
        self.planer = planer

    def _sample_trajectory(self,
                           dynamic_system: Union[ExactDynamicSystem],
                           unroll_length: int,
                           num_unrolls: int,
                           state_sample_ranges,
                           control_sample_ranges=None,
                           noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                           parameter_changer: ParameterChanger = ConstantParameter()
                           ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        if not isinstance(parameter_changer, ConstantParameter):
            raise NotImplemented("ControlledTrajectorySampler only works with ConstantParameter, because ExactEnv "
                                 "does not support parameter changes")
        if isinstance(dynamic_system, ExactDynamicSystem):
            raise NotImplemented("ExactDynamicSystem are not supported until now.")
        else:
            raise Exception("dynamic_system is not a known type.")
        return x, y

    def _sample_trajectory_env(self,
                               dynamic_system: Union[ExactDynamicSystem],
                               unroll_length: int,
                               num_unrolls: int,
                               state_sample_ranges,
                               control: Float[torch.Tensor, "batch time control"],
                               noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                               parameter_changer: ParameterChanger = ConstantParameter()
                               ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:

        assert noise_adder.process_context_noise_level == 0 and noise_adder.process_target_noise_level == 0, \
            "Envs do not support process noise."

        random_init = dynamic_system.sample_init_state(state_sample_ranges[self.state_sample_type], num_unrolls)
        random_init = random_init.reshape((num_unrolls, -1))

        dynamic_system.reset(init_state=random_init,
                             parameter_changer=parameter_changer)
        self.controller.reset_with_given_data(random_init, dynamic_system.parameters)

        y_all = []
        y = random_init
        for t in tqdm(range(unroll_length + 1)):
            plan = self.planer.plan(y, dynamic_system.get_goal())
            control = self.controller.obtain_sol(y, plan, t)
            x = dynamic_system.step(control, t)[0]
            y = noise_adder.add_observation_noise(x[None])[0]
            y_all.append(y)

        y_t = torch.stack(y_all, dim=0).permute(1, 0, 2)
        y_tp1 = y_t[:, 1:]
        y_t = y_t[:, :-1]
        y_t = torch.concatenate([y_t, control[:, 1:, :]], dim=-1)
        return y_t, y_tp1

class CombinedSampler(TrajectorySampler):

    def __init__(self, sampler: List[TrajectorySampler], prop: List[float]):
        """

        Args:
            sampler: List of TrajectorySampler
            prop: Probability in which the sampler is selected.
        """
        super().__init__()
        self.sampler = sampler
        self.prop = prop

    def _sample_trajectory(self,
                           dynamic_system: Union[ExactDynamicSystem],
                           unroll_length: int,
                           num_unrolls: int,
                           state_sample_ranges,
                           control_sample_ranges=None,
                           noise_adder: ProcessAndObservationNoiseAdder = ZeroNoiseAdder(),
                           parameter_changer: ParameterChanger = ConstantParameter()
                           ) -> \
            Tuple[Float[torch.Tensor, "batch time in"], Float[torch.Tensor, "batch time out"]]:
        chosen_sampler: TrajectorySampler = np.random.choice(self.sampler, p=self.prop)
        return chosen_sampler._sample_trajectory(
            dynamic_system,
            unroll_length,
            num_unrolls,
            state_sample_ranges,
            control_sample_ranges,
            noise_adder,
            parameter_changer)
