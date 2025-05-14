"""
trainer.py 定义通用的训练器类

    @Time    : 2025/05/09
    @Author  : JackWang
    @File    : trainer.py
    @IDE     : VsCode
    @Copyright Copyright Shihong Wang (c) 2025 with GNU Public License V3.0
"""

# Standard Library
import copy
import random
from pathlib import Path
from datetime import datetime
from typing import Any, Optional

# Third-Party Library
from omegaconf import DictConfig, OmegaConf

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# My Library
from .helper import (
    to,
    get_dtype,
    get_device,
    get_dataset,
    get_optimizer,
    get_scheduler,
    get_transforms,
)
from .logger import RichuruLogger

from algorithms.base import (
    WeaklySupervisedSemanticSegmentationAlgorithm as WSSSAlgorithm,
)


class Trainer:
    def __init__(self, algorithm: WSSSAlgorithm, config: DictConfig):

        self.setup_config(config)

        self.epoch = 0
        self.best_epoch = 0
        self.best_metrics: dict[str, float] = {"mIoU": 0.0}
        self.best_losses: dict[str, float] = {"val_loss": float("inf")}

        self.device = get_device()
        self.dtype = get_dtype(config.data.dtype)

        self.algorithm: WSSSAlgorithm = to(algorithm, self.dtype, self.device)

        self.log_dir = (
            Path(__file__).resolve().parents[1]
            / f"logs/{ self.config.algorithm.name}/{datetime.now():%Y%m%d-%H:%M:%S}/"
        )
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.setup_datasets()
        self.setup_optimizer()
        self.setup_scheduler()
        self.load_checkpoint(config.checkpoint.load_file)
        self.setup_tensorboard(log_dir=self.log_dir)
        self.setup_logger()
        self.setup_algorithm()

    def __del__(self):
        self.config.running_info["finished_at"] = f"{datetime.now():%Y%m%d-%H:%M:%S}"
        self.save_config(self.log_dir / "training_config.yaml")
        self.writer.close()

    def setup_config(self, config: DictConfig):
        config.running_info = {
            "training_at": None,
            "finished_at": None,
            "start_at": f"{datetime.now():%Y%m%d-%H%M%S}",
        }
        self.config = config

    def setup_algorithm(self):

        self.algorithm.logger = self.logger
        self.algorithm.writer = self.writer

    def setup_datasets(self):

        self.val_transform = get_transforms(self.config.val.transform)
        self.train_transform = get_transforms(self.config.train.transform)

        self.valset, self.trainset = get_dataset(
            self.config.data.dataset_using, (self.val_transform, self.train_transform)
        )

        self.val_loader, self.train_loader = [
            DataLoader(dataset, **cfg)
            for dataset, cfg in zip(
                [self.valset, self.trainset],
                (self.config.val.dataloader, self.config.train.dataloader),
            )
        ]

    def setup_optimizer(self):
        self.optimizer = get_optimizer(
            self.algorithm.model.parameters(), self.config.algorithm.optimizer
        )

    def setup_scheduler(self):
        self.scheduler = get_scheduler(self.optimizer, self.config.algorithm.scheduler)

    def setup_tensorboard(self, log_dir: Path):
        self.writer = None
        smd = log_dir / "tensorboard"
        smd.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(smd)

    def setup_logger(self):
        self.logger = RichuruLogger(
            epochs=(self.config.train.epochs - self.epoch),
            batches=len(self.train_loader),
            log_dir=self.log_dir,
        )

    def load_checkpoint(self, load_file: Optional[str | Path] = None):
        if load_file is None:
            self.kwargs = {}
            return

        checkpoint = torch.load(load_file, map_location=self.device)
        self.algorithm.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])

        self.epoch = checkpoint["epoch"]
        self.kwargs = checkpoint["kwargs"]
        self.best_epoch = checkpoint["best_epoch"]
        self.best_losses |= checkpoint["best_losses"]
        self.best_metrics |= checkpoint["best_metrics"]

    def save_checkpoint(
        self, save_dir: Path, is_best: bool = False, kwargs: dict[str, Any] = None
    ):
        if kwargs is None:
            kwargs = {}
        else:
            self.kwargs.update(kwargs)

        save_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "algorithm": self.algorithm.__class__.__name__,
            "model": self.algorithm.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
            "best_epoch": self.best_epoch,
            "best_losses": self.best_losses,
            "best_metrics": self.best_metrics,
            "kwargs": self.kwargs,
        }

        # current epoch state dict
        if getattr(self, "first_epoch", None) is None:
            self.logger.success("Training checkpoints is saved to logs/checkpoints")
            self.first_epoch = False

        # save every N epochs
        if (
            self.config.checkpoint.save_every is not None
            and self.epoch % self.config.checkpoint.save_every == 0
        ):
            torch.save(
                checkpoint,
                save_dir / f"{save_dir.stem}_e{self.epoch}.pth",
            )

        # save best model
        if is_best:
            torch.save(
                checkpoint,
                save_dir / f"{save_dir.stem}_best.pth",
            )
            torch.save(
                self.get_model_state_dict(),
                save_dir / f"{save_dir.stem}_model.pth",
            )

    def save_config(self, save_path: Optional[str | Path] = None):
        save_path.parent.mkdir(exist_ok=True, parents=True)
        OmegaConf.save(self.config, save_path)

    def get_model_state_dict(self):
        self.algorithm.model.to("cpu")
        model = copy.deepcopy(self.algorithm.model)
        self.algorithm.model.to(self.device)
        return model.state_dict()

    def train_epoch(self):
        for batch_idx, batch_data in enumerate(self.train_loader):
            batch_data = to(batch_data, self.dtype, self.device)

            self.optimizer.zero_grad()
            returns = self.algorithm.train_step(batch_data, self.epoch, batch_idx)
            returns["loss"].backward()
            self.optimizer.step()

            if batch_idx % self.config.train.log_interval == 0:
                self.logger.info(
                    f"Epoch [{self.epoch:{len(str(self.epoch))}}] "
                    f"Batch [{batch_idx:{len(str(len(self.train_loader)))}}/{len(self.train_loader)}], "
                    f"Loss {returns['loss']:.4f}"
                )

            self.logger.update_batch(self.algorithm.get_info_dict())

            if batch_idx == 10:
                break

    @torch.no_grad()
    def validate_epoch(self):
        return {"val_loss": random.random()}, {"mIoU": random.random()}

    def train(self):
        self.config.running_info["training_at"] = f"{datetime.now():%Y%m%d-%H:%M:%S}"

        with self.logger.training_context():
            for epoch in range(self.epoch, self.config.train.epochs):
                self.epoch = epoch
                self.train_epoch()
                val_losses, val_metrics = self.validate_epoch()

                self.scheduler.step()

                is_best = val_losses["val_loss"] < self.best_losses["val_loss"]
                if is_best:
                    self.best_epoch = self.epoch
                    self.best_losses |= val_metrics
                    self.best_metrics |= val_metrics
                self.save_checkpoint(
                    self.log_dir / "checkpoints",
                    is_best=is_best,
                    kwargs={"epoch": self.epoch},
                )

                self.logger.update_epoch(self.algorithm.get_info_dict())
