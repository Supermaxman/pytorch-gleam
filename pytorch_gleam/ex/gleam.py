import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import CALLBACK_REGISTRY, DATAMODULE_REGISTRY, LightningCLI

import pytorch_gleam.callbacks
import pytorch_gleam.data.datasets


def main():
    CALLBACK_REGISTRY.register_classes(pytorch_gleam.callbacks, pl.Callback)
    DATAMODULE_REGISTRY.register_classes(pytorch_gleam.data.datasets, pl.LightningDataModule)
    LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        run=True,
        subclass_mode_model=True,
        subclass_mode_data=True,
        auto_registry=True,
    )


if __name__ == "__main__":
    main()
