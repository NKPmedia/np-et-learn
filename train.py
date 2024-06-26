import pyrootutils
import comet_ml
import torch
from lightning_fabric import seed_everything
from src.utils.other import is_debugging
from jaxtyping import install_import_hook

if is_debugging():
    hook = install_import_hook("src", "typeguard.typechecked")
from src.utils.other import extras, get_metric_value, task_wrapper, is_debugging
from src.utils.pylogger import get_pylogger, log_hyperparameters

root = pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
import os

os.environ["root"] = str(root)
os.environ["PROJECT_ROOT"] = str(root)
from typing import List, Optional

import hydra

from omegaconf import DictConfig
from pytorch_lightning import Trainer, LightningDataModule, Callback, LightningModule
from pytorch_lightning.loggers import Logger

from src.utils.instantiators import instantiate_loggers, instantiate_callbacks
if is_debugging():
    hook.uninstall()
log = get_pylogger(__name__)


@task_wrapper
def train(cfg: DictConfig):
    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("compile"):
        log.info("Compiling model!")
        model = torch.compile(model)

    if cfg.get("yappi-profile"):
        import yappi
        yappi.start()

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    if cfg.get("yappi-profile"):
        yappi.stop()
        stats = yappi.get_func_stats()
        stats.save(os.path.join(cfg.paths.log_dir, "yappi/run.callgrind"), type='callgrind')
    train_metrics = trainer.callback_metrics

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict

@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
