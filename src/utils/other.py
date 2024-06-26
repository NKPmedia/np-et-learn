import gzip
import os.path
import warnings
from importlib.util import find_spec
from pathlib import Path
from typing import List, Callable, Sequence

import numpy as np
import rich
import rich.syntax
import rich.tree
import torch
from hydra.core.hydra_config import HydraConfig
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import DictConfig, OmegaConf, open_dict
from rich.prompt import Prompt
from torch import Tensor
from tqdm.contrib.concurrent import process_map

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


from multiprocessing import Pool
import glob

import pickle


def make_short(path):
    from src.systems.runner.predict_test_runner import ShortEnvRunResult, EnvRunResult

    short_path = path+".short"
    if os.path.exists(short_path):
        return
    with gzip.open(path, "rb") as f:
        results: List[EnvRunResult] = pickle.load(f)
        short_results = []
        for res in results:
            short_results.append(
                ShortEnvRunResult(
                    delta_t = res.delta_t,
                    score = res.control.score
                )
            )
        del results
        #save short results
        with gzip.open(short_path, "wb") as sf:
            pickle.dump(short_results, sf)


def make_short_pred(path):
    from src.systems.runner.predict_test_runner import ShortPredEnvRunResult, EnvRunResult

    short_path = path+".short"
    if os.path.exists(short_path):
        return
    with gzip.open(path, "rb") as f:
        results: List[EnvRunResult] = pickle.load(f)
        short_results = []
        for res in results:
            short_results.append(
                ShortPredEnvRunResult(
                    delta_t = res.delta_t,
                    score=res.control.score,
                    std=res.prediction.std,
                    errorToSystem=res.prediction.error.toSystem,
                    cost=res.control.cost
                )
            )
        del results
        #save short results
        with gzip.open(short_path, "wb") as sf:
            pickle.dump(short_results, sf)

def create_reward_env_results(path: str, cfg: DictConfig):
    paths = []
    root = os.path.join(cfg.paths.log_dir, path)

    # find all pkl.gzip files in subdirectories of root
    for path in glob.glob(os.path.join(root, "**/*.pkl.gzip"), recursive=True):
        #check if path ends with .pkl.gzip
        if path.endswith(".pkl.gzip"):
            paths.append(path)

    process_map(make_short, paths)

def create_short_pred_env_results(path: str, cfg: DictConfig):
    paths = []
    root = os.path.join(cfg.paths.log_dir, path)

    # find all pkl.gzip files in subdirectories of root
    for path in glob.glob(os.path.join(root, "**/*.pkl.gzip"), recursive=True):
        # check if path ends with .pkl.gzip
        if path.endswith(".pkl.gzip"):
            paths.append(path)

    process_map(make_short_pred, paths)

def get_eval_log_path(cfg: DictConfig, path) -> str:
    """
    joins the path with the eval log path
    if there is a "get_latest_folder" in the path it will take the latest folder at this place.
    The folder is assumed to be a timestamp
    If there are multiple folders it will print a warning and take the newest one
    """
    if "get_latest_folder" in path:
        path = path.replace("get_latest_folder", "*")
        folders = list(Path(cfg.paths.log_dir).glob(path))
        if len(folders) > 1:
            log.warning(
                f"Multiple folders found for {path}. Taking the newest one. All folders: {folders}"
            )
            print(f"Multiple folders found for {path}. Taking the newest one. All folders: {folders}")
        if len(folders) == 0:
            raise ValueError(f"No folder found for {path}")
        path = str(sorted(folders)[-1])
    return os.path.join(cfg.paths.log_dir, path)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def var2std(data: Tensor):
    """
    Converts tensor from variance to standard deviation form.
    Args:
        data:

    Returns:
    """
    return torch.sqrt(data)


def std2var(data: Tensor):
    """
    Converts tensor from standard deviation to variance form.
    Args:
        data:

    Returns:
    """
    return data ** 2


def get_context_index_sweep(grid_size: int, context_numbers_add: List[int]):
    context_indeeces = []
    for context_number_add in context_numbers_add:
        indices = list(range(grid_size))
        if len(context_indeeces) > 0:
            for elem in context_indeeces[-1]:
                if elem in indices:
                    indices.remove(elem)
            context_indeeces.append(
                context_indeeces[-1] + np.random.choice(indices, context_number_add, replace=False).tolist())
        else:
            context_indeeces.append(np.random.choice(indices, context_number_add, replace=False).tolist())
    return context_indeeces


def extras(cfg: DictConfig) -> None:
    """Applies optional utilities before the task is started.
    Utilities:
    - Ignoring python warnings
    - Setting tags from command line
    - Rich config printing
    """

    # return if no `extras` config
    if not cfg.get("extras"):
        log.warning("Extras config not found! <cfg.extras=null>")
        return

    # disable python warnings
    if cfg.extras.get("ignore_warnings"):
        log.info("Disabling python warnings! <cfg.extras.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # prompt user to input tags from command line if none are provided in the config
    if cfg.extras.get("enforce_tags"):
        log.info("Enforcing tags! <cfg.extras.enforce_tags=True>")
        enforce_tags(cfg, save_to_file=True)

    # pretty print config tree using Rich library
    if cfg.extras.get("print_config"):
        log.info("Printing config tree with Rich! <cfg.extras.print_config=True>")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper(task_func: Callable) -> Callable:
    """Optional decorator that controls the failure behavior when executing the task function.
    This wrapper can be used to:
    - make sure loggers are closed even if the task function raises an exception (prevents multirun failure)
    - save the exception to a `.log` file
    - mark the run as failed with a dedicated file in the `logs/` folder (so we can find and rerun it later)
    - etc. (adjust depending on your needs)
    Example:
    ```
    @utils.task_wrapper
    def train(cfg: DictConfig) -> Tuple[dict, dict]:
        ...
        return metric_dict, object_dict
    ```
    """

    def wrap(cfg: DictConfig):
        # execute the task
        try:
            metric_dict, object_dict = task_func(cfg=cfg)

        # things to do if exception occurs
        except Exception as ex:
            # save exception to `.log` file
            log.exception("")

            # some hyperparameter combinations might be invalid or cause out-of-memory errors
            # so when using hparam search plugins like Optuna, you might want to disable
            # raising the below exception to avoid multirun failure
            raise ex

        # things to always do after either success or exception
        finally:
            # display output dir path in terminal
            log.info(f"Output dir: {cfg.paths.output_dir}")

            # always close wandb run (even if exception occurs so multirun won't fail)
            if find_spec("wandb"):  # check if wandb is installed
                import wandb

                if wandb.run:
                    log.info("Closing wandb!")
                    wandb.finish()

        return metric_dict, object_dict

    return wrap


def get_metric_value(metric_dict: dict, metric_name: str) -> float:
    """Safely retrieves value of the metric logged in LightningModule."""

    if not metric_name:
        log.info("Metric name is None! Skipping metric value retrieval...")
        return None

    if metric_name not in metric_dict:
        raise Exception(
            f"Metric value not found! <metric_name={metric_name}>\n"
            "Make sure metric name logged in LightningModule is correct!\n"
            "Make sure `optimized_metric` name in `hparams_search` config is correct!"
        )

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric value! <{metric_name}={metric_value}>")

    return metric_value


@rank_zero_only
def print_config_tree(
        cfg: DictConfig,
        print_order: Sequence[str] = (
                "data",
                "model",
                "callbacks",
                "logger",
                "trainer",
                "paths",
                "extras",
        ),
        resolve: bool = False,
        save_to_file: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        cfg (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
        save_to_file (bool, optional): Whether to export config to the hydra output folder.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    queue = []

    # add fields from `print_order` to queue
    for field in print_order:
        queue.append(field) if field in cfg else log.warning(
            f"Field '{field}' not found in config. Skipping '{field}' config printing..."
        )

    # add all the other fields to queue (not specified in `print_order`)
    for field in cfg:
        if field not in queue:
            queue.append(field)

    # generate config tree from queue
    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = cfg[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    # print config tree
    rich.print(tree)

    # save config tree to file
    if save_to_file:
        with open(Path(cfg.paths.output_dir, "config_tree.log"), "w") as file:
            rich.print(tree, file=file)


@rank_zero_only
def enforce_tags(cfg: DictConfig, save_to_file: bool = False) -> None:
    """Prompts user to input tags from command line if no tags are provided in config."""

    if not cfg.get("tags"):
        if "id" in HydraConfig().cfg.hydra.job:
            raise ValueError("Specify tags before launching a multirun!")

        log.warning("No tags provided in config. Prompting user to input tags...")
        tags = Prompt.ask("Enter a list of comma separated tags", default="dev")
        tags = [t.strip() for t in tags.split(",") if t != ""]

        with open_dict(cfg):
            cfg.tags = tags

        log.info(f"Tags: {cfg.tags}")

    if save_to_file:
        with open(Path(cfg.paths.output_dir, "tags.log"), "w") as file:
            rich.print(cfg.tags, file=file)


def is_debugging():
    import sys

    gettrace = getattr(sys, 'gettrace', None)

    if gettrace is None:
        print('No sys.gettrace')
    elif gettrace():
        print('Hmm, Big Debugger is watching me')
        return True
    else:
        return False
