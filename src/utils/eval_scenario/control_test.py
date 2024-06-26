import gzip
import os
import pickle
from typing import List

import torch
from tqdm import tqdm

from src.data.dataset.base import BasePckHdf5Loader
from src.modules.base_np import BaseNP
from src.systems.control.controller import Controller
from src.systems.enviroment.invP_env import InvertedPendulumEnv
from src.systems.enviroment.vdp_env import VdpEnv
from src.systems.planner.planner import NotPlanner
from src.systems.runner.control_runner import ControlRunner
from src.systems.runner.interactors import Interactor
from src.systems.runner.predict_test_runner import EnvRunResult
from src.systems.systems.NNModel import NNModel
from src.systems.systems.dynamic_system import DynamicSystemType
from src.utils.eval_scenario.eval_scenario import EvalScenario
from src.utils.other import chunks
from src.utils.visualization.run_plotter import MultiRunPlotter


class VdpControlTestScenario(EvalScenario):

    def __init__(self,
                 controller: Controller,
                 interactor: Interactor,
                 control_interactor: Interactor,
                 observation_noise: float,
                 process_noise: float,
                 system_type: DynamicSystemType,
                 chunk_size: int = 300,
                 **kwargs):
        super().__init__(**kwargs)
        self.observation_noise = observation_noise
        self.process_noise = process_noise
        self.controller = controller
        self.interactor = interactor
        self.control_interactor = control_interactor
        self.chunk_size = chunk_size
        self.system_type = system_type
        self.results = []

    def compare_results(self, files: List[str], names: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            files: The files containing the results of the model.

        Returns:

        """
        runs = []
        for file in files:
            with open(file, "rb") as f:
                results: List[EnvRunResult] = pickle.load(f)
            runs.append(results)

        MultiRunPlotter.plot_error_over_time(runs, labels=names)

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, device: str = "cuda",
             one_step_prediction_models: List[BaseNP] = None,
             video_path: str = None):

        model = NNModel(loaded_model=model, system_type=self.system_type) if isinstance(model, BaseNP) else model
        if one_step_prediction_models is None:
            one_step_prediction_nn_models = None
        else:
            one_step_prediction_nn_models = []
            for one_step_model in one_step_prediction_models:
                one_step_prediction_nn_models.append(
                    NNModel(loaded_model=one_step_model, system_type=self.system_type) if isinstance(one_step_model,
                                                                                                 BaseNP) else one_step_model)

        with torch.no_grad():
            env = VdpEnv(pre_generated_data=data,
                         observation_noise=self.observation_noise,
                         process_noise=self.process_noise,
                         interactor=self.interactor)
            self.controller.set_model(model, one_step_prediction_nn_models)
            plan = NotPlanner(50)
            runner = ControlRunner(env, self.controller, planner=plan, device=device, print_steps=False, control_interactor=self.control_interactor)
            for chunk in tqdm(chunks(range(len(data)), self.chunk_size), total=len(data) // self.chunk_size):
                res = runner.run(list(chunk))
                self.results.extend(res)

    def save_result(self, dir_path: str):
        with gzip.open(os.path.join(dir_path, f"control_test_{self.name}_results.pkl.gzip"), "w", compresslevel=4) as f:
            pickle.dump(self.results, f)


class InvPControlTestScenario(EvalScenario):

    def __init__(self,
                 controller: Controller,
                 interactor: Interactor,
                 control_interactor: Interactor,
                 state_interactor: Interactor,
                 observation_noise: float,
                 process_noise: float,
                 system_type: DynamicSystemType,
                 chunk_size: int = 300,
                 log_attention: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.observation_noise = observation_noise
        self.process_noise = process_noise
        self.interactor = interactor
        self.control_interactor = control_interactor
        self.state_interactor = state_interactor
        self.controller = controller
        self.chunk_size = chunk_size
        self.system_type = system_type
        self.log_attention = log_attention
        self.results = []

    def compare_results(self, files: List[str], names: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            files: The files containing the results of the model.

        Returns:

        """
        runs = []
        for file in files:
            with open(file, "rb") as f:
                results: List[EnvRunResult] = pickle.load(f)
            runs.append(results)

        MultiRunPlotter.plot_error_over_time(runs, labels=names)

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, device: str = "cuda", one_step_prediction_models: List[BaseNP] = None,
             video_path: str = None):

        model = NNModel(loaded_model=model, system_type=self.system_type) if isinstance(model, BaseNP) else model
        one_step_prediction_nn_models = []
        if one_step_prediction_models is None:
            one_step_prediction_nn_models = None
        else:
            for one_step_model in one_step_prediction_models:
                one_step_prediction_nn_models.append(
                    NNModel(loaded_model=one_step_model, system_type=self.system_type) if isinstance(one_step_model,
                                                                                                 BaseNP) else one_step_model)

        with torch.no_grad():
            env = InvertedPendulumEnv(pre_generated_data=data,
                         observation_noise=self.observation_noise,
                         process_noise=self.process_noise,
                         interactor=self.interactor)
            self.controller.set_model(model, one_step_prediction_nn_models)
            plan = NotPlanner(50)
            runner = ControlRunner(env, self.controller, planner=plan, device=device, print_steps=False, control_interactor=self.control_interactor,
                                   log_attention=self.log_attention,
                                   state_interactor=self.state_interactor)
            for chunk in tqdm(chunks(range(len(data)), self.chunk_size), total=len(data) // self.chunk_size):
                res = runner.run(list(chunk))
                self.results.extend(res)

    def save_result(self, dir_path: str):
        with gzip.open(os.path.join(dir_path, f"control_test_{self.name}_results.pkl.gzip"), "w", compresslevel=4) as f:
            pickle.dump(self.results, f)

class GymControlTestScenario(EvalScenario):

    def __init__(self,
                 controller: Controller,
                 interactor: Interactor,
                 observation_noise: float,
                 process_noise: float,
                 system_type: DynamicSystemType,
                 env_name: str,
                 chunk_size: int = 300,
                 render_env: str = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.observation_noise = observation_noise
        self.process_noise = process_noise
        self.interactor = interactor
        self.controller = controller
        self.chunk_size = chunk_size
        self.system_type = system_type
        self.env_name = env_name
        self.render_env = render_env
        self.results = []

    def compare_results(self, files: List[str], names: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            files: The files containing the results of the model.

        Returns:

        """
        runs = []
        for file in files:
            with open(file, "rb") as f:
                results: List[EnvRunResult] = pickle.load(f)
            runs.append(results)

        MultiRunPlotter.plot_error_over_time(runs, labels=names)

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, device: str = "cuda", one_step_prediction_models: List[BaseNP] = None,
             video_path: str = None):

        model = NNModel(loaded_model=model, system_type=self.system_type) if isinstance(model, BaseNP) else model
        if one_step_prediction_models is None:
            one_step_prediction_nn_models = None
        else:
            one_step_prediction_nn_models = []
            for one_step_model in one_step_prediction_models:
                one_step_prediction_nn_models.append(NNModel(loaded_model=one_step_model, system_type=self.system_type) if isinstance(one_step_model, BaseNP) else one_step_model)

        assert video_path is not None or self.render_env is None, "If render_env is not None, video_path has to be provided!"

        with torch.no_grad():
            env = GymEnv(pre_generated_data=data,
                         observation_noise=self.observation_noise,
                         process_noise=self.process_noise,
                         interactor=self.interactor,
                         backend="mujoco",
                         env_name=self.env_name,
                         system_type=self.system_type,
                         render_env=self.render_env,
                         video_base_path=video_path
                         )
            self.controller.set_model(model, one_step_prediction_nn_models)
            plan = NotPlanner(30)
            runner = ControlRunner(env, self.controller, planner=plan, device=device, print_steps=False)
            for chunk in tqdm(chunks(range(len(data)), self.chunk_size), total=len(data) // self.chunk_size):
                res = runner.run(list(chunk))
                self.results.extend(res)

    def save_result(self, dir_path: str):
        with gzip.open(os.path.join(dir_path, f"control_test_{self.name}_results.pkl.gzip"), "w", compresslevel=4) as f:
            pickle.dump(self.results, f)