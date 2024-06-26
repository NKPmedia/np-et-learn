import bz2
import gzip
import pickle
from functools import partial
from math import sqrt, log
from typing import List, Sequence, Tuple, Union

import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from tabulate import tabulate
from torch import Tensor
from torch.nn.functional import l1_loss

from src.data.dataset.base import BasePckHdf5Loader
from src.systems.predictor.reset.reset_trigger import ResetTrigger, PlotableResetTrigger
from src.systems.runner.predict_test_runner import EnvRunResult, SystemResult, PredictErrorResult, PredictionResult, \
    ControlResult
from src.systems.systems.dynamic_system import DynamicSystemType
from src.utils.visualization.dynamic_system import visualize_nd_trajectories
from src.utils.visualization.plot_helpers import plot_with_std, plot_channel_subplot
from multiprocessing.pool import ThreadPool
import mgzip

from src.utils.visualization.visualization_utils import initialize_plot, load_colors

c = load_colors()

# load params
params = initialize_plot('README')  # specifies font size etc., adjust accordingly
plt.rcParams.update(params)

class RunPlotter:

    def plot_predictor_data(self, res: EnvRunResult, system: DynamicSystemType, number_plot_cols: int = 2):
        """
        Plots the data from one predictor test run
        Plots the trajectory, the prediction, the error to the observed state and the error to the real state
        Args:
            res:
            system:

        Returns:

        """
        trajectories = self._get_trajectories(res)
        visualize_nd_trajectories(trajectories=trajectories, trajectory_names=["system", "observed", "prediction"],
                                  channel_names=system.state_names, number_plot_cols=number_plot_cols,
                                  title="Trajectory")

        self._plot_control(res, system, number_plot_cols=number_plot_cols)

        self._plot_abs_pred_error(res, system, number_plot_cols=number_plot_cols)

        self._plot_error_distribution(res, system, number_plot_cols=number_plot_cols)

    def _get_trajectories(self, res: EnvRunResult) -> \
            Float[Tensor, "trajectory time state"]:
        """
        Returns the true and observed trajectories of the system and the prediction
        Args:
            res:

        Returns:

        """
        trajectries = torch.stack([res.system.system_states, res.system.observed_states, res.prediction.system_states])
        return trajectries

    def _plot_control(self, res: EnvRunResult, system: DynamicSystemType, number_plot_cols: int = 2):
        """
        Plots the control of the system
        Uses one plot and a sublot for each control
        Uses number of controls and control names from the system type
        Args:
            res:
            system:

        Returns:

        """
        visualize_nd_trajectories(
            trajectories=res.system.controls[None],
            trajectory_names=["control"],
            channel_names=system.control_names,
            delta_t=0.02,
            number_plot_cols=number_plot_cols,
            title="Control input"
        )

    def _plot_abs_pred_error(self, res, system, number_plot_cols: int = 2):
        """
        Plots the error of the prediction
        Uses one plot and a sublot for each state
        Uses number of states and state names from the system type
        Args:
            res:
            system:

        """
        error = l1_loss(res.system.system_states, res.prediction.system_states, reduce=False)
        # calculate moving average of error with window size 10. Mage is as long as error
        smooth10_error = torch.nn.functional.avg_pool1d(error.T, kernel_size=10, stride=1, padding=5).T[:-1]

        visualize_nd_trajectories(
            trajectories=torch.concatenate([error[None], smooth10_error[None]], dim=0),
            trajectory_names=["error", "Avg 10"],
            channel_names=system.state_names,
            delta_t=0.02,
            number_plot_cols=number_plot_cols,
            title="Absolute prediction error"
        )

    def _plot_error_distribution(self, res, system, number_plot_cols):
        """
        Plot a histogram of the absolute prediction error
        Plots the mean in the histogram
        Args:
            res:
            system:
            number_plot_cols:

        Returns:

        """
        error = l1_loss(res.system.system_states, res.prediction.system_states, reduce=False)
        error = error.mean(dim=-1)
        mean_error = error.mean()
        fig = plt.figure()
        fig.gca().hist(error, bins=20, color='c', edgecolor='k', alpha=0.65)
        fig.gca().axvline(mean_error, color='k', linestyle='dashed', linewidth=1)

        min_ylim, max_ylim = plt.ylim()
        fig.gca().text(mean_error * 1.1, max_ylim * 0.9, 'Mean: {:.5f}'.format(mean_error))
        fig.show()


class MultiRunPlotter:

    def __init__(self, type: DynamicSystemType):
        self.type = type
        self.labels = []

    def plot_error(self, results: List[EnvRunResult], type: DynamicSystemType):
        pass

    def set_labels(self, labels: List[str]):
        self.labels = labels

    def plot_error_over_time(self,
                             results: List[List[EnvRunResult]],
                             labels: List[str] = None,
                             delta_t: float = 0.02,
                             std: bool = True,
                             move_avg: int = 1,
                             number_plot_cols: int = 2,
                             x_axis_limits: Sequence[float] = None,
                             y_axis_limits: Sequence[float] = None,
                             normalize: bool = False,
                             exlude_time_steps: List[int] = []):
        """
        Plots the mean error and the plot_std over time for multiple runs
        Drops the first 2 steps because the window is empty (no real prediction!)
        Args:
            delta_t:
            labels:
            results:

        Returns:

        """

        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0)-len(exlude_time_steps))) * delta_t
        data = []
        for run, label in zip(results, labels):
            errors = [res.prediction.error.toSystem for res in run]
            errors = torch.stack(errors)
            #exclude time steps
            if exlude_time_steps is not None:
                mask = torch.ones(errors.shape[1], dtype=torch.bool)
                mask[exlude_time_steps] = False
                errors = errors[:, mask]
            errors = errors[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            if normalize:
                inc_std = torch.tensor(self.type.inc_std)
                errors = errors / inc_std
            mean = errors.mean(dim=-1)
            print(f"Total mean error for {label}: {errors.mean()}")
            errors = torch.concatenate([errors, mean[:, :, None]], dim=-1)
            data.append(errors)

        subtitles = self.type.state_names + ("mean",)
        data = torch.stack(data)
        fig = plot_channel_subplot(time, data, labels,
                                   subtitles=subtitles,
                                   number_plot_cols=number_plot_cols,
                                   std=std,
                                   move_avg=move_avg,
                                   x_axis_limits=x_axis_limits,
                                   y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("Prediction error over time")
        fig.show()

    def plot_cum_error_over_time(self,
                                 results: List[List[EnvRunResult]],
                                 labels: List[str] = None,
                                 delta_t: float = 0.02,
                                 std: bool = True,
                                 move_avg: int = 1,
                                 number_plot_cols: int = 2,
                                 x_axis_limits: Sequence[float] = None,
                                 y_axis_limits: Sequence[float] = None,
                                 normalize: bool = False):
        """
        Plots the mean error and the plot_std over time for multiple runs
        Drops the first 2 steps because the window is empty (no real prediction!)
        Args:
            delta_t:
            labels:
            results:

        Returns:

        """
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            errors = [res.prediction.error.toSystem for res in run]
            errors = torch.stack(errors)
            if normalize:
                inc_std = torch.tensor(self.type.inc_std)
                errors = errors / inc_std
            errors = errors[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            errors_cum = torch.cumsum(errors, dim=1)
            mean = errors_cum.mean(dim=-1)
            errors = torch.concatenate([errors_cum, mean[:, :, None]], dim=-1)
            data.append(errors)

        subtitles = self.type.state_names + ("mean",)
        data = torch.stack(data)
        fig = plot_channel_subplot(time, data, labels,
                                   subtitles=subtitles,
                                   number_plot_cols=number_plot_cols,
                                   std=std,
                                   move_avg=move_avg,
                                   x_axis_limits=x_axis_limits,
                                   y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("Cumulated prediction error over time")
        fig.show()

    def _load_single_pkl(self, file, run_idx: List[int] = None):
        if file.endswith(".pkl"):
            with open(file, "rb") as f:
                results: List[EnvRunResult] = pickle.load(f)
        elif file.endswith(".pkl.bz2"):
            data = bz2.BZ2File(file, "rb")
            results: List[EnvRunResult] = pickle.load(data)
        elif file.endswith(".pkl.gzip") or file.endswith(".pkl.gzip.short"):
            with gzip.open(file, "rb") as f:
                results: List[EnvRunResult] = pickle.load(f)
        else:
            raise ValueError(f"Unknown file ending: {file}")
        if run_idx is not None:
            results = [results[i] for i in run_idx]
        return results

    def load_pkl(self, files, run_idx: List[int] = None):
        """Load the pkls over multiple threads"""
        import multiprocessing
        with ThreadPool(processes=4) as pool:
            partial_load = partial(self._load_single_pkl, run_idx=run_idx)
            results = pool.map(partial_load, files)
        return results

    def plot_likelihood_over_time(self,
                                  results: List[List[EnvRunResult]],
                                  labels: Sequence[str] = None,
                                  delta_t: float = 0.02,
                                  move_avg: int = 1,
                                  std: bool = True,
                                  number_plot_cols: int = 2
                                  ):
        """
        Plots the likelihood over time for multiple runs
        Args:
            runs:
            labels:

        Returns:

        """
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            pred_mean = torch.stack([res.prediction.system_states for res in run])
            pred_std = torch.stack([res.prediction.std for res in run])
            gt_state = torch.stack([res.system.system_states for res in run])
            distribution = torch.distributions.Normal(pred_mean, pred_std)
            log_prop = distribution.log_prob(gt_state)
            likelihoods = torch.exp(log_prop)
            likelihoods = likelihoods[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            mean = likelihoods.mean(dim=-1)
            likelihoods = torch.concatenate([likelihoods, mean[:, :, None]], dim=-1)
            data.append(likelihoods)

        subtitles = self.type.state_names + ("mean",)
        data = torch.stack(data)
        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols, std=std,
                                   move_avg=move_avg)
        fig.legend()
        fig.suptitle("Real state likelihood based on prediction")
        fig.show()

    def plot_pred_latent_std_over_time(self,
                                       results,
                                       labels: Sequence[str] = None,
                                       delta_t: float = 0.02,
                                       move_avg: int = 1,
                                       std: bool = True,
                                       x_axis_limits: Sequence[float] = None,
                                       y_axis_limits: Sequence[float] = None,
                                       plot_mean: bool = True
                                       ):
        if not plot_mean and len(results) > 1:
            raise ValueError("Plotting the mean over episods of mean latent std of multiple runs is not supported!")
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            pred_std = torch.stack([res.prediction.latent_std for res in run])
            pred_std = pred_std[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            mean = pred_std.mean(dim=-1)[:, :, None]
            data.append(mean)

        data = torch.stack(data)
        if plot_mean:
            subtitles = ("mean",)

            fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=1, std=std,
                                       move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits)
            fig.legend()
            fig.suptitle("Prediction latent std over time")
            fig.show()
        else:
            #plot for each episode
            label = labels[0]
            run = data[0, :,:, 0]
            print(run.shape)
            fig = plt.figure(figsize=(12, 6))
            fig.gca().set_title(label)
            fig.gca().set_xlabel("Time [s]")
            fig.gca().set_ylabel("Latent std")
            for i in range(run.size(0)):
                fig.gca().plot(time, data[0, i,:, 0], alpha=0.5)

    def plot_reset_trigger(self,
                           results: List[List[EnvRunResult]],
                           trigger: PlotableResetTrigger,
                           labels: Sequence[str] = None,
                           delta_t: float = 0.02,
                             ):

        labels = self.get_labels(labels)
        data = []
        bound_data = []
        assert len(results) == 1, "Only one run set is supported"
        assert len(results[0]) == 1, "Only one episode is supported"

        prediction = results[0][0].prediction.system_states
        observation = results[0][0].system.observed_states
        std = results[0][0].prediction.std
        prediction = prediction[None, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
        std = std[None, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
        observation = observation[None, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
        trigger.reset(batch_size=1)

        triggers = []
        bounds = []
        observed_value = []
        for t in range(prediction.size(1) - 1):
            trig = trigger.test_trigger(pred_states=prediction[:, :t + 1],
                                        pred_std=std[:, :t + 1],
                                        observed_states=observation[:, :t + 1],
                                        step=t)
            bounds.append(trigger.get_bound(pred_states=prediction[:, :t + 1],
                                        pred_std=std[:, :t + 1],
                                        observed_states=observation[:, :t + 1],
                                        step=t)
                          )
            observed_value.append(trigger.get_observed_values(pred_states=prediction[:, :t + 1],
                                                              pred_std=std[:, :t + 1],
                                                              observed_states=observation[:, :t + 1],
                                                              step=t)
                          )

            triggers.append(trig)
        triggers = torch.stack(triggers, dim=1)
        bounds = torch.stack(bounds, dim=1)
        observed_value = torch.stack(observed_value, dim=1)
        num_ros = observed_value.shape[-1]
        # Plot a subplot with num_ros rows and plot the observed value and the bound
        fig, axs = plt.subplots(num_ros, 1, figsize=(12, 6))
        if num_ros == 1:
            axs = [axs]
        for i in range(num_ros):
            axs[i].plot(observed_value[0, :, i], label="observed value")
            axs[i].plot(bounds[0, :, i], label="bound", linestyle="--", color="black")
        # find the idy of the trigger points
        trigger_points = torch.nonzero(triggers[0]).flatten()
        for trigger_point in trigger_points:
            for i in range(num_ros):
                axs[i].axvline(x=trigger_point, color="red", alpha=0.5)
        fig.show()


    def plot_pred_std_over_time(self,
                                results,
                                labels: Sequence[str] = None,
                                delta_t: float = 0.02,
                                move_avg: int = 1,
                                std: bool = True,
                                number_plot_cols: int = 2,
                                x_axis_limits: Sequence[float] = None,
                                y_axis_limits: Sequence[float] = None
                                ):
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            pred_std = torch.stack([res.prediction.std for res in run])
            pred_std = pred_std[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            mean = pred_std.mean(dim=-1)
            pred_std = torch.concatenate([pred_std, mean[:, :, None]], dim=-1)
            data.append(pred_std)

        subtitles = self.type.state_names + ("mean",)
        data = torch.stack(data)
        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols, std=std,
                                   move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("Prediction std over time")
        fig.show()

    def plot_abs_pred_error_with_likelihoodbound(self,
                                                 results: List[List[EnvRunResult]],
                                                 labels: Sequence[str] = None,
                                                 delta_t: float = 0.02,
                                                 prop: float = 0.999,
                                                 min_simulatious_trigger: int = 1,
                                                 bound: str = "gauss_bound",
                                                 number_plot_cols: int = 2,
                                                 x_axis_limits: Sequence[float] = None,
                                                 y_axis_limits: Sequence[float] = None,
                                                 size: Tuple[int, int] = None,
                                                 channels: Sequence[int] = None,
                                                 plot_red_line: bool = True,
                                                 title: bool = True,
                                                 time_adaption: bool = False):
        """
        Plots the error of the prediction to the observed state
        Uses one plot and a sublot for each state
        Uses number of states and state names from the system type

        Website for bounds: https://www.probabilitycourse.com/chapter6/6_2_5_jensen's_inequality.php
        https://www.stat.cmu.edu/~arinaldo/Teaching/36709/S19/Scribed_Lectures/Jan24_Nil-Jana.pdf
        https://en.wikipedia.org/wiki/Moment-generating_function


        Args:
            res:
            system:

        """
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        bound_data = []
        for run, label in zip(results, labels):
            errors = [res.system.observed_states - res.prediction.system_states for res in run]
            errors = torch.stack(errors)
            std = [res.prediction.std for res in run]
            std = torch.stack(std)
            errors = errors[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            std = std[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            if bound == "gauss_bound":
                bound_size = sqrt(2) * std * torch.erfinv(torch.Tensor([prop])).item()
                trigger_val = errors
            elif bound in ["gauss_chernoff", "gauss_chernoff_7in10"]:
                prop_out = 1 - prop
                time_step = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0)))
                time_step = time_step[None, :, None].tile((std.shape[0], 1, std.shape[-1]))
                pi_t = torch.pi**2 * time_step**2 / 6
                if not time_adaption:
                    pi_t = torch.ones_like(pi_t)
                bound_size = torch.sqrt(-2 * std ** 2 * torch.log(1/pi_t*torch.Tensor([prop_out / 2])))
                trigger_val = errors
            elif bound == "hoeffding_bound":
                length = 5
                prop_out = 1 - prop
                std_squared = std ** 2
                moving_sum_window_std_squared = torch.cumsum(std_squared, dim=1) - \
                                                torch.nn.functional.pad(torch.cumsum(std_squared, dim=1)[:, :-length],
                                                                        pad=(0, 0, length, 0), value=0)
                tmp_ones = torch.ones_like(moving_sum_window_std_squared)
                sum_length = torch.cumsum(tmp_ones, dim=1) - \
                             torch.nn.functional.pad(torch.cumsum(tmp_ones, dim=1)[:, :-length], pad=(0, 0, length, 0),
                                                     value=0)
                sum_length = sum_length[:, :, 0, None]

                bound_size = torch.sqrt(-(2 * log(prop_out / 2) * moving_sum_window_std_squared) / (sum_length ** 2))

                abs_error_sum = torch.abs(torch.cumsum(errors, dim=1) - \
                                          torch.nn.functional.pad(torch.cumsum(errors, dim=1)[:, :-length],
                                                                  pad=(0, 0, length, 0), value=0))
                trigger_val = abs_error_sum / sum_length

            else:
                raise ValueError("Unknown bound type!")
            data.append(trigger_val)
            bound_data.append(bound_size)

        data = torch.stack(data).mean(dim=1)
        bound_data = torch.stack(bound_data).mean(dim=1)

        if channels is None:
            channel = data.size(-1)
        else:
            channel = len(channels)
            data = data[:, :, channels]
            bound_data = bound_data[:, :, channels]
        rows = int(np.ceil(channel / number_plot_cols))
        if size is None:
            size = (12 * number_plot_cols, rows * 3)
        fig = plt.figure(figsize=size)
        trigger_channels = torch.zeros((data.size(0), data.size(1)))
        for i in range(channel):
            ax = fig.add_subplot(rows, number_plot_cols, i + 1)
            if title:
                ax.set_title(self.type.state_names[i])
            if x_axis_limits is not None:
                ax.set_xlim(x_axis_limits)
            if y_axis_limits is not None:
                ax.set_ylim(y_axis_limits)
            for j in range(data.size(0)):
                if i == 0:
                    ax.plot(time, data[j, :, i], label=labels[j], linewidth=0.8)
                else:
                    ax.plot(time, data[j, :, i], linewidth=0.8)
                ax.plot(time, bound_data[j, :, i], "--", color="black")
                if data.min() < 0:
                    ax.plot(time, -bound_data[j, :, i], "--", color="black")

                # marks the trigger points
                if bound == "gauss_chernoff_7in10":
                    window = 10
                    threshold = 7
                    # Checks where the data is above the bound for 3 out of 5 steps
                    trigger_points = torch.abs(data[j, :, i]) > bound_data[j, :, i]
                    trigger_points = np.where(trigger_points)[0]
                    for trigger_point in trigger_points:

                        if trigger_point + (window - 1) < len(time):
                            in_window_trigger_points = (torch.abs(data[j, trigger_point:trigger_point + window, i])
                                                        > bound_data[j, trigger_point:trigger_point + window, i])
                            if in_window_trigger_points.sum() >= threshold:
                                ax.axvline(x=time[trigger_point], color="red", alpha=0.5)
                else:
                    # Plot a green vertical line where  abs data is above the bound
                    trigger_points = torch.abs(data[j, :, i]) > bound_data[j, :, i]
                    trigger_points = np.where(trigger_points)[0]
                    trigger_channels[j, trigger_points] += 1
                    for trigger_point in trigger_points:
                        ax.axvline(x=time[trigger_point], color="green", alpha=0.2)
        for i in range(channel):
            # ith axis
            ax = fig.axes[i]
            # plot red line where the trigger is above the threshold
            assert trigger_channels.size(0) == 1, "Only one run is supported"
            trigger_points = trigger_channels[0] >= min_simulatious_trigger
            trigger_points = np.where(trigger_points)[0]
            if plot_red_line:
                for trigger_point in trigger_points:
                    ax.axvline(x=time[trigger_point], color="red", alpha=0.5)

        legend = fig.legend(loc='upper right', bbox_to_anchor=(0.9, 0.9))
        legend.get_frame().set_alpha(None)
        fig.tight_layout()
        if title:
            fig.suptitle(f"Prediction error with bound (P: {prop}) based on predicted std")
        fig.show()
        fig.savefig(f"/tmp/pred_error_with_bound_{bound}_{prop}.pdf")


    def give_trigger_points(self,
                     results: List[List[EnvRunResult]],
                     trigger: ResetTrigger,
                     labels: Sequence[str] = None,
                     delta_t: float = 0.02,
                     ):
        """
        Plots the error of the prediction to the observed state
        Uses one plot and a sublot for each state
        Uses number of states and state names from the system type

        Website for bounds: https://www.probabilitycourse.com/chapter6/6_2_5_jensen's_inequality.php
        https://www.stat.cmu.edu/~arinaldo/Teaching/36709/S19/Scribed_Lectures/Jan24_Nil-Jana.pdf
        https://en.wikipedia.org/wiki/Moment-generating_function


        Args:
            res:
            system:

        """
        labels = self.get_labels(labels)
        data = []
        bound_data = []
        assert len(results) == 1, "Only one run set is supported"
        for run in results:
            prediction = [res.prediction.system_states for res in run]
            observation = [res.system.observed_states for res in run]
            std = [res.prediction.std for res in run]
            std = torch.stack(std)
            prediction = torch.stack(prediction)
            observation = torch.stack(observation)
            prediction = prediction[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            std = std[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            observation = observation[:, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
            trigger.reset(batch_size=prediction.size(0))

            triggers = []
            for t in range(prediction.size(1)-1):
                trig = trigger.test_trigger(pred_states=prediction[:, :t+1],
                                     pred_std=std[:, :t+1],
                                     observed_states=observation[:, :t+1],
                                     step=t)
                triggers.append(trig)
            triggers = torch.stack(triggers, dim=1)
            print(triggers.shape)
        return triggers

    def plot_trigger_points(self,
                            results: List[List[EnvRunResult]],
                            labels: Sequence[str] = None,
                            delta_t: float = 0.02,
                            number_plot_cols: int = 2):
        """
        Plots the error of the prediction
        Uses one plot and a sublot for each state
        Uses number of states and state names from the system type
        Args:
            res:
            system:

        """
        labels = self.get_labels(labels)
        fig = plt.figure(figsize=(12, 6))

        for run, label in zip(results, labels):
            trigger_times = []
            run_number = []
            for i, res in enumerate(run):
                trigger_points = res.prediction.trigger.flatten()
                # get the idx where trigger_points is True
                trigger_points = torch.nonzero(trigger_points).flatten()
                # get the time of the trigger points
                trigger_time = trigger_points * delta_t
                trigger_times.extend(trigger_time.tolist())
                run_number.extend([i for _ in range(trigger_time.size(0))])
            fig.gca().scatter(trigger_times, run_number, label=label, marker='x', alpha=0.2)
            # set the x axis range to 0 to the length of the run
            fig.gca().set_xlim([0, results[0][0].system.system_states.size(0) * delta_t])
        fig.legend()
        fig.suptitle(f"Trigger positions")
        fig.show()

    def plot_window_length_points(self,
                                  results: List[List[EnvRunResult]],
                                  labels: Sequence[str] = None,
                                  delta_t: float = 0.02,
                                  x_axis_limits: Sequence[float] = None,
                                  y_axis_limits: Sequence[float] = None,
                                  aspect: float = 1):
        """
        Plots the window length as an image plot. The color indicates the window length
        Args:
            results:
            labels:
            delta_t:

        Returns:

        """

        labels = self.get_labels(labels)
        # For more trajectorie plot a heatmap for each run
        if len(results[0]) > 1:
            for run, label in zip(results, labels):
                window_sizes = []
                for i, res in enumerate(run):
                    window_sizes.append(res.prediction.window_size)
                image = torch.stack(window_sizes)
                fig = plt.figure(figsize=(12, 6))
                if x_axis_limits is not None:
                    fig.gca().set_xlim(x_axis_limits)
                if y_axis_limits is not None:
                    fig.gca().set_ylim(y_axis_limits)
                fig.gca().imshow(image, cmap='hot', aspect=aspect)
                fig.suptitle(f"Window length for {label}")
                fig.show()
        # For one trajectory plot the window size over time
        else:
            time = torch.arange(1 * delta_t, results[0][0].prediction.error.toSystem.size(0) * delta_t, delta_t)
            for run, label in zip(results, labels):
                res = run[0]
                if x_axis_limits is not None:
                    plt.gca().set_xlim(x_axis_limits)
                if y_axis_limits is not None:
                    plt.gca().set_ylim(y_axis_limits)
                plt.plot(time, res.prediction.window_size)
            plt.title("Window length over time")
            plt.show()

    def plot_state_value(self,
                         results: List[List[EnvRunResult]],
                         labels: Sequence[str] = None,
                         delta_t: float = 0.02,
                         number_plot_cols: int = 2,
                         x_axis_limits: Sequence[float] = None,
                         y_axis_limits: Sequence[float] = None,
                         plot_observation: bool = False):
        """
        Plots the state value of the runs
        Args:
            results: The results of the simulations for several runs
            x_axis_limits: The limits of the x axis
            y_axis_limits: The limits of the y axis
            plot_observation: Plot the observation in addition to the state value
        """
        assert len(results) == 1, "Only one run set is supported"
        labels = self.get_labels(labels)
        assert len(labels) == 1, "Only one label is supported"
        labels = labels + ["Observations"]
        time = torch.arange(0, results[0][0].prediction.error.toSystem.size(0) * delta_t, delta_t)
        control = [res.system.controls for res in results[0]]
        state = [res.system.system_states for res in results[0]]
        observation = [res.system.observed_states for res in results[0]]
        control = torch.stack(control)
        state = torch.stack(state)
        observation = torch.stack(observation)

        subtitles = self.type.state_names + self.type.control_names

        state_control = torch.concatenate([state, control], dim=-1)[None]
        if plot_observation:
            zero_control = torch.zeros_like(control)
            observation_control = torch.concatenate([observation, zero_control], dim=-1)[None]
            data = torch.concatenate([state_control, observation_control], dim=0)
        else:
            data = state_control

        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols,
                                   std=False,
                                   move_avg=1, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("States and control over time")
        fig.show()

    def plot_control(self,
                     results: List[List[EnvRunResult]],
                     labels: Sequence[str] = None,
                     delta_t: float = 0.02,
                     number_plot_cols: int = 2,
                     move_avg: int = 1,
                     x_axis_limits: Sequence[float] = None,
                     y_axis_limits: Sequence[float] = None,
                     std: bool = True):
        """
        Plots the control of the runs
        Args:
            results: The results of the simulations for several runs
            labels: The names of the runs
            std: Plot the standard deviation of the costs over the simulations

        Returns:
        """
        labels = self.get_labels(labels)
        time = torch.arange(0, results[0][0].prediction.error.toSystem.size(0) * delta_t, delta_t)
        data = []
        for run, label in zip(results, labels):
            control = torch.stack([res.system.controls for res in run])
            data.append(control)

        subtitles = self.type.control_names
        data = torch.stack(data)
        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols, std=std,
                                   move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("Control over time")
        fig.show()

    def plot_costs(self,
                   results: List[List[EnvRunResult]],
                   labels: Sequence[str] = None,
                   delta_t: float = 0.02,
                   number_plot_cols: int = 2,
                   move_avg: int = 1,
                   std: bool = True,
                   x_axis_limits: Sequence[float] = None,
                   y_axis_limits: Sequence[float] = None,
                   **kwargs):
        """
        Plots the cost of the runs (basically how well the controller did) over time
        Args:
            results: The results of the simulations for several runs
            labels: The names of the runs
            std: Plot the standard deviation of the costs over the simulations

        Returns:
        """
        labels = self.get_labels(labels)
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            cost = torch.stack([res.control.cost for res in run])
            cost = cost[:,
                   2:]  # remove first and second steps because the costs can sometime only be calculated after the first step
            data.append(cost)

        subtitles = ("total",) + self.type.cost_names
        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols, std=std,
                                   move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits,
                                   **kwargs)
        fig.legend()
        fig.suptitle("Costs over time")
        fig.show()

    def plot_cum_costs(self,
                       results: List[List[EnvRunResult]],
                       labels: Sequence[str] = None,
                       delta_t: float = 0.02,
                       number_plot_cols: int = 2,
                       move_avg: int = 1,
                       std: bool = True,
                       x_axis_limits: Sequence[float] = None,
                       y_axis_limits: Sequence[float] = None,
                       time_normalization: bool = False,
                       **kwargs):
        """
        Plots the cumulated cost of the runs (basically how well the controller did) over time
        Args:
            results: The results of the simulations for several runs
            labels: The names of the runs
            std: Plot the standard deviation of the costs over the simulations

        Returns:
        """
        labels = self.get_labels(labels)

        # Print control scores first
        # Print it as a table with tabulate
        score_table = []
        for run, label in zip(results, labels):
            scores = torch.stack([res.control.score for res in run])
            score_table.append([label, torch.mean(scores), torch.std(scores)])
        print(tabulate(score_table, headers=["Run", "Score mean ", "Std"], tablefmt="github"))

        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        data = []
        for run, label in zip(results, labels):
            cost = torch.stack([res.control.cost for res in run])
            cost = cost[:,
                   2:]  # remove first and second steps because the costs can sometime only be calculated after the first step
            cum_cost = torch.cumsum(cost, dim=1)
            if time_normalization:
                cum_cost = cum_cost / time[None, :, None]
            data.append(cum_cost)

        subtitles = ("total",) + self.type.cost_names
        fig = plot_channel_subplot(time, data, labels, subtitles=subtitles, number_plot_cols=number_plot_cols,
                                   std=std,
                                   move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits,
                                   **kwargs)
        fig.legend()
        fig.suptitle("Costs over time")
        fig.show()

    def plot_cum_costs_per_run(self,
                               results: List[List[EnvRunResult]],
                               labels: Sequence[str] = None):
        """
        Plots the cumulated costs of the episodes as a table with tabulate
        Rows are the episodes, columns are the costs per experiment/run
        In addition a bar chart is plotted with a bar for each episode. The experiments have different colors.
        Args:
            results:
            labels:

        Returns:

        """
        labels = self.get_labels(labels)

        score_table = []
        episode_number = len(results[0])
        for i in range(episode_number):
            episode_costs = []
            for run, label in zip(results, labels):
                score_mean = run[i].control.score[0]
                episode_costs.append(score_mean)
            score_table.append(episode_costs)
        print(tabulate(score_table, headers=labels, tablefmt="github"))

        # Plot the costs as a groupd bar chart
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.set_title("Cumulated costs per episode")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulated costs")
        x = np.arange(episode_number)
        width = 1 / len(labels)
        for i, run in enumerate(results):
            score = torch.stack([res.control.score for res in run])
            score = score[:, 0]
            ax.bar(x + i * width
                   , score, width, label=labels[i])
        ax.legend()
        ax.xaxis.set_major_locator(plt.MaxNLocator(20))
        fig.show()

    def plot_parameter(self, data: BasePckHdf5Loader,
                       run_idx: int,
                       subtitles: Sequence[str],
                       delta_t: float = 0.02,
                       number_plot_cols: int = 2,
                       move_avg: int = 1,
                       std: bool = True,
                       x_axis_limits: Sequence[float] = None,
                       y_axis_limits: Sequence[float] = None,
                       **kwargs):
        parameter = torch.tensor(data["parameter"][run_idx])

        time = torch.arange(0, parameter.size(0) * delta_t, delta_t)

        fig = plot_channel_subplot(time, parameter[None, None], ["Parameter value"], subtitles=subtitles,
                                   number_plot_cols=number_plot_cols,
                                   std=std,
                                   move_avg=move_avg, x_axis_limits=x_axis_limits, y_axis_limits=y_axis_limits,
                                   **kwargs)
        fig.legend()
        fig.suptitle("Parameter over time")
        fig.show()

    def print_trigger_statistics(self, results: List[List[EnvRunResult]], labels: Sequence[str] = None):

        labels = self.get_labels(labels)

        # Print a tabel with the 10 runs with the most trigger events
        trigger_table = []
        for run, label in zip(results, labels):
            trigger = torch.stack([res.prediction.trigger for res in run])
            trigger_count = torch.sum(trigger, dim=-1)
            # Get the 10 runs with the most trigger events
            trigger_count, indices = torch.topk(trigger_count, 10)
            trigger_table.append([label, trigger_count.tolist(), indices.tolist()])
        print(tabulate(trigger_table, headers=["Run", "Trigger count", "Indices"], tablefmt="github"))

    def get_labels(self, labels: Union[Sequence[str], None]):
        if labels is None:
            assert self.labels is not None, "No labels given and no labels saved in the object"
            labels = self.labels
        return labels

    def plot_diff_pred_error(self,
                             results: List[List[EnvRunResult]],
                             labels: List[str] = None,
                             delta_t: float = 0.02,
                             std: bool = True,
                             move_avg: int = 1,
                             number_plot_cols: int = 2,
                             x_axis_limits: Sequence[float] = None,
                             y_axis_limits: Sequence[float] = None,
                             normalize: bool = False):
        labels = self.get_labels(labels)
        assert len(results) == 1, "Only one run set is supported"
        time = torch.tensor(range(2, results[0][0].prediction.error.toSystem.size(0))) * delta_t
        errors = [res.prediction.all_system_states - res.system.system_states
                  for res in results[0]]
        errors = torch.stack(errors).abs()
        errors = errors[:, :, 2:]  # remove first 2 steps because the window is empty (no real prediction!)
        if normalize:
            inc_std = torch.tensor(self.type.inc_std)
            errors = errors / inc_std
        mean = errors.mean(dim=-1)
        errors = torch.concatenate([errors, mean[..., None]], dim=-1).permute(1, 0, 2, 3)

        subtitles = self.type.state_names + ("mean",)
        fig = plot_channel_subplot(time, errors, labels,
                                   subtitles=subtitles,
                                   number_plot_cols=number_plot_cols,
                                   std=std,
                                   move_avg=move_avg,
                                   x_axis_limits=x_axis_limits,
                                   y_axis_limits=y_axis_limits)
        fig.legend()
        fig.suptitle("Prediction error over time")
        fig.show()
