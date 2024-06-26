import os.path
from glob import glob
from logging import Logger
from typing import List, Tuple

import hydra
import pyrootutils
from lightning_fabric import seed_everything

from src.systems.systems.dynamic_system import ExactDynamicSystem

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer

from src.utils.instantiators import instantiate_loggers
from src.utils.other import task_wrapper, extras
from src.utils.pylogger import get_pylogger, log_hyperparameters

log = get_pylogger(__name__)


@task_wrapper
def evaluate(cfg: DictConfig) -> Tuple[dict, dict]:
    """Evaluates given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)
    else:
        log.warning("No seed provided!")
        print("WARNING: No seed provided!")

    if not cfg.get("exact_model"):
        assert cfg.ckpt_path, "No checkpoint config provided!"
        if cfg.ckpt_path in ["checkpoint-epoch", "last"]:
            files = glob(os.path.join(cfg.paths.experiment_dir, "**", f"{cfg.ckpt_path}*.ckpt"), recursive=True)
            if len(files) != 1:
                raise Exception(f"If now checkpoint the search has to find exactly one but found: {files}")
            cfg.ckpt_path = files[0]
        elif "/*/" in cfg.ckpt_path:
            files = glob(os.path.join(cfg.paths.experiment_dir, cfg.ckpt_path), recursive=True)
            if len(files) != 1:
                raise Exception(f"If now checkpoint the search has to find exactly one but found: {files}")
            cfg.ckpt_path = files[0]

    log.info(f"Instantiating testdataset <{cfg.eval_data.data._target_}>")
    data = hydra.utils.instantiate(cfg.eval_data.data)

    log.info(f"Instantiating eval scenario <{cfg.eval_data.scenario._target_}>")
    scenario = hydra.utils.instantiate(cfg.eval_data.scenario)

    if not cfg.get("exact_model"):
        log.info(f"Instantiating model <{cfg.model._target_}>")
        model: LightningModule = hydra.utils.instantiate(cfg.model)
        model = model.__class__.load_from_checkpoint(cfg.ckpt_path, strict=cfg.strict_load, map_location="cpu")
    else:
        model: ExactDynamicSystem = hydra.utils.instantiate(cfg.model)

    if cfg.get("one_step_prediction_models"):
        one_step_prediction_models = []

        for one_step_prediction_model_name, model_path in zip(cfg.one_step_prediction_models.keys(), cfg.one_step_prediction_model_paths):
            if "/*/" in model_path:
                files = glob(os.path.join(cfg.paths.experiment_dir, model_path), recursive=True)
                if len(files) != 1:
                    raise Exception(f"If now checkpoint the search has to find exactly one but found: {files}")
                model_path = files[0]
            log.info(f"Instantiating one step prediction model <{cfg.one_step_prediction_models[one_step_prediction_model_name]._target_}>")
            one_step_prediction_model: LightningModule = hydra.utils.instantiate(cfg.one_step_prediction_models[one_step_prediction_model_name])
            one_step_prediction_model = one_step_prediction_model.load_from_checkpoint(model_path, strict=cfg.strict_load, map_location="cpu")
            one_step_prediction_models.append(one_step_prediction_model)
    else:
        one_step_prediction_models = None

    object_dict = {
        "cfg": cfg,
        "data": data,
        "scenario": scenario,
        "model": model,
    }

    log.info("Starting eval!")
    scenario.eval(model, data, cfg.device, one_step_prediction_models=one_step_prediction_models,
                  video_path=os.path.join(cfg.paths.output_dir, "videos"))

    scenario.save_result(cfg.paths.output_dir)

    return {}, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> None:
    print("Main func")
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    evaluate(cfg)


if __name__ == "__main__":
    print("Starting eval...")
    main()
