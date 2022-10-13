import os
from typing import Optional

import pytorch_lightning as pl
import yaml
from clearml import Task
from pytorch_lightning.callbacks import Callback


class ClearMLTask(Callback):
    def __init__(self, project_name: str):
        super().__init__()
        self.project_name = project_name

    def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: Optional[str] = None) -> None:
        task_name = os.path.basename(trainer.logger.save_dir)
        self.task = Task.init(project_name=self.project_name, task_name=task_name)
        config_path = os.path.join(trainer.logger.save_dir, "config.yaml")
        self.task.connect_configuration(config_path)
        with open(config_path) as f:
            config = yaml.load(f)
            self.task.connect(config)
