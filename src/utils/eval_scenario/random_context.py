import os
from typing import List

import torch
from torch.nn.functional import l1_loss
from tqdm import tqdm

from src.data.dataset.base import BasePckHdf5Loader
from src.data.noise_adders import NoiseAdder
from src.modules.base_np import BaseNP
from src.utils.eval_scenario.eval_scenario import EvalScenario


class RandomContextScenario(EvalScenario):

    def __init__(self, noise_adder: NoiseAdder, state_idx, **kwargs):
        super().__init__(**kwargs)
        self.noise_adder = noise_adder
        self.state_idx = state_idx
        self.results = {
            "mean": [],
            "std": [],
            "c_length": [],
        }

    def compare_results(self, data: BasePckHdf5Loader, files: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            data: The ground truth data.
            files: The files containing the results of the model.

        Returns:

        """
        l1_mean = []
        l1 = []
        for file in files:
            result = torch.load(file)
            mean = result["mean"]
            std = result["std"]
            c_length = torch.stack(result["c_length"])[..., 0]
            # Get all possible context lengths
            c_lengths = torch.unique(c_length)

            l1_file_mean = {length.item(): [] for length in c_lengths}
            l1_file = {length.item(): [] for length in c_lengths}

            for length in c_lengths:
                for run in range(len(mean)):
                    mae = l1_loss(mean[run][c_length[run] == length, :, :], data.db["y"][c_length[run] == length, :, :],
                                  reduce=False)
                    mae_mean = torch.mean(mae, dim=(2))
                    l1_file_mean[length.item()].append(mae)
                    l1_file[length.item()].append(mae_mean)
            l1_mean.append(l1_file_mean)
            l1.append(l1_file)
        return l1_mean, l1

    def compare_results_logpdf(self, data: BasePckHdf5Loader, files: List[str]):
        """
        Compares the result of some model with the ground truth data.
        Args:
            data: The ground truth data.
            files: The files containing the results of the model.

        Returns:

        """
        logpdf_data = []
        for file in files:
            result = torch.load(file)
            mean = result["mean"]
            std = result["var"]
            runs = []
            for run in range(len(mean)):
                logpdf = torch.distributions.Normal(mean[run], std[run]).log_prob(data.db["y"])
                logpdf = torch.mean(logpdf, dim=(2))
                runs.append(logpdf)
            logpdf_data.append(runs)
        return logpdf_data

    def eval(self, model: BaseNP, data: BasePckHdf5Loader, device, runs: int = 3, **kwargs):
        with torch.no_grad():
            model.eval()
            model.to(device)
            data.to(device)
            idx_chunk = data.idx_data["idx_chunk"]
            context_idx = data.idx_data["context_idx"]
            c_length = data.idx_data["c_length"]

            for run in range(runs):
                pred_means = torch.empty_like(data.db["y"], device=torch.device("cpu"))
                pred_stds = torch.empty_like(data.db["y"], device=torch.device("cpu"))
                c_lengths = torch.empty((data.db["y"].size(0), 1), device=torch.device("cpu"))
                for chunk_nr, chunk in enumerate(tqdm(idx_chunk[run])):
                    gt_unroll_length = data.db["x"].size(1)
                    context_start = context_idx[run][chunk_nr][0]
                    context_end = context_idx[run][chunk_nr][1]
                    context_x = data.db["x_context"][chunk, context_start:context_end]
                    context_y = data.db["y_context"][chunk, context_start:context_end]
                    context_x, context_y = self.noise_adder.add_noise_context(context_x, context_y)

                    dist, _, _, _ = model.forward(context_x,
                                                        context_y,
                                                        data.db["x"][chunk])
                    pred_means[chunk] = dist.mean.cpu()[:]
                    pred_stds[chunk] = dist.stddev.cpu()[:]
                    c_lengths[chunk] = c_length[run][chunk_nr]

                self.results["mean"].append(pred_means)
                self.results["std"].append(pred_stds)
                self.results["c_length"].append(c_lengths)

    def save_result(self, dir_path: str):
        torch.save(self.results, os.path.join(dir_path, f"random_context_{self.name}_results.pt"))
