from clearml import Task
from pytorch_lightning.callbacks import Callback


class ClearMLTask(Callback):
    def __init__(self, project_name: str, task_name: str):
        super().__init__()
        self.project_name = project_name
        self.task_name = task_name
        self.task = Task.init(project_name=self.project_name, task_name=self.task_name)
