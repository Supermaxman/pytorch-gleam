from pytorch_lightning.callbacks.progress import RichProgressBar
from pytorch_lightning.callbacks.progress.rich_progress import CustomProgress
from rich.console import Console


class TPURichProgressBar(RichProgressBar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console: Console = Console()
            self._console.clear_live()
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                refresh_per_second=self._refresh_rate_per_second if self._refresh_rate_per_second > 0 else 1,
                disable=self._enabled and self._refresh_rate_per_second > 0,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False
