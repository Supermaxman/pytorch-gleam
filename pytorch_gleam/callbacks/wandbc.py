import os

import pytorch_lightning as pl
import wandb
import yaml
from pytorch_lightning.callbacks import Callback


class WandbConfig(Callback):
    def __init__(self):
        super().__init__()
        self.loaded_config = False

    def _init(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.loaded_config:
            return
        config_path = os.path.join(trainer.logger.save_dir, "config.yaml")
        with open(config_path) as f:
            config = yaml.safe_load(f)
            wandb.config.update(config)
        self.loaded_config = True

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the train begins."""
        self._init(trainer, pl_module)

    def on_validation_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the validation loop begins."""
        self._init(trainer, pl_module)

    def on_test_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the test begins."""
        self._init(trainer, pl_module)

    def on_predict_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Called when the predict begins."""
        self._init(trainer, pl_module)
