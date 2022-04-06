import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import _TPU_AVAILABLE
from pytorch_lightning.utilities.exceptions import MisconfigurationException

if _TPU_AVAILABLE:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met


class XLAGraphMonitor(Callback):
    def __init__(self):
        super().__init__()
        if not _TPU_AVAILABLE:
            raise MisconfigurationException("Cannot use XLAGraphMonitor when TPUs are not available")

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx,
        unused=0,
    ) -> None:
        xm.master_print(met.metrics_report())
        xm.master_print(met.metric_names())
        xm.master_print(met.counter_names())
