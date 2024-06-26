import gzip
import os
import pickle
from typing import List

import torch
from torch.nn.functional import l1_loss
from tqdm import tqdm

from src.data.dataset.base import BasePckHdf5Loader
from src.modules.base_np import BaseNP
from src.systems.enviroment.trajectory_replay_env import TrajectoryReplay
from src.systems.predictor.predictor import Predictor
from src.systems.runner.predict_test_runner import PredictTestRunner, EnvRunResult
from src.systems.systems.NNModel import NNModel
from src.systems.systems.dynamic_system import DynamicSystemType
from src.utils.eval_scenario.eval_scenario import EvalScenario
from src.utils.other import chunks
from src.utils.visualization.run_plotter import MultiRunPlotter


class PredictTestScenario(EvalScenario):

    def __init__(self, predictor: Predictor, observation_noise: float, system_type: DynamicSystemType, chunk_size: int = 300, **kwargs):
        super().__init__(**kwargs)
        self.observation_noise = observation_noise
        self.predictor = predictor
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

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, device: str = "cuda", one_step_prediction_models = None,
             video_path = None):
        assert one_step_prediction_models is None
        model = NNModel(loaded_model=model, system_type=self.system_type) if isinstance(model, BaseNP) else model

        with torch.no_grad():
            self.predictor.set_model(model)
            env = TrajectoryReplay(data["x"],
                                   self.system_type.state_size,
                                   observation_noise=self.observation_noise)
            #predictor = MovingWindowNNPredictor(model, window_size=300, reset_points=[600])
            runner = PredictTestRunner(env, self.predictor, device=device, print_steps=False, parallel_runs=self.chunk_size)
            for chunk in tqdm(chunks(range(len(data)), self.chunk_size), total=len(data) // self.chunk_size):
                res = runner.run(list(chunk))
                self.results.extend(res)

    def save_result(self, dir_path: str):
        with gzip.open(os.path.join(dir_path, f"predictor_test_{self.name}_results.pkl.gzip"), "w", compresslevel=5) as f:
            pickle.dump(self.results, f)