import os
from typing import Optional

import pytorch_lightning as pl
import yaml
from pytorch_lightning.callbacks import Callback


class ClearMLTask(Callback):
    def __init__(self, project_name: str):
        super().__init__()
        self.project_name = project_name
        self.initialized = False
        self.loaded_config = False

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        """Called when fit, validate, test, predict, or tune begins."""
        if self.initialized:
            return
        task_name = os.path.basename(trainer.default_root_dir)
        from clearml import Task

        self.task = Task.init(project_name=self.project_name, task_name=task_name)
        self.initialized = True

    def _init(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        if self.loaded_config:
            return
        config_path = os.path.join(
            trainer.logger.save_dir, trainer.logger.name, f"version_{trainer.logger.version}", "config.yaml"
        )
        self.task.connect_configuration(config_path)
        with open(config_path) as f:
            config = yaml.safe_load(f)
            self.task.connect(config)
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
