import os
from typing import List

import torch
from torch.nn.functional import l1_loss
from tqdm import tqdm

from src.data.dataset.base import BasePckHdf5Loader
from src.data.noise_adders import NoiseAdder
from src.modules.base_np import BaseNP
from src.utils.eval_scenario.eval_scenario import EvalScenario


class FixedContextScenario(EvalScenario):

    def __init__(self, noise_adder: NoiseAdder, start_idx: List[int], end_idx: List[int], state_idx, **kwargs):
        super().__init__(**kwargs)
        self.noise_adder = noise_adder
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.state_idx = state_idx
        self.results = {
            "mean": [],
            "plot_std": []
        }

    def compare_results(self, data: BasePckHdf5Loader, files: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            data: The ground truth data.
            files: The files containing the results of the model.

        Returns:

        """
        l1 = []
        for file in files:
            result = torch.load(file)
            mean = result["mean"]
            std = result["plot_std"]
            runs = []
            for run in range(len(mean)):
                mae = l1_loss(mean[run], data.db["y"], reduce=False)
                mae = torch.mean(mae, dim=(2))
                runs.append(mae)
            l1.append(runs)
        return l1

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, one_step_prediction_model = None):
        assert one_step_prediction_model is None
        with torch.no_grad():
            model.eval()
            model.to("cuda")
            data.to("cuda")
            idx_chunk = data.idx_data["idx_chunk"]
            for start_i, end_i in zip(self.start_idx, self.end_idx):
                pred_means = torch.empty_like(data.db["y"], device=torch.device("cpu"))
                pred_stds = torch.empty_like(data.db["y"], device=torch.device("cpu"))
                for chunk_nr, chunk in enumerate(tqdm(idx_chunk[0])):
                    gt_unroll_length = data.db["x"].size(1)
                    context_x = data.db["x_context"][chunk, start_i:end_i]
                    context_y = data.db["y_context"][chunk, start_i:end_i]
                    control_size = context_x.size(2)-context_y.size(2)
                    context_x, context_y = self.noise_adder.add_noise_context(context_x, context_y)
                    control = torch.zeros((context_x.size(0), gt_unroll_length, control_size), device=context_x.device)
                    pred_mean, pred_std = model.unroll_forward(context_x,
                                                               context_y,
                                                               data.db["x"][chunk, 0, None][..., self.state_idx],
                                                               control=control)
                    pred_means[chunk] = pred_mean.cpu()[:, 1:]
                    pred_stds[chunk] = pred_std.cpu()[:, 1:]

                self.results["mean"].append(pred_means)
                self.results["plot_std"].append(pred_stds)

    def save_result(self, dir_path: str):
        torch.save(self.results, os.path.join(dir_path, f"fixed_context_{self.name}_results.pt"))
