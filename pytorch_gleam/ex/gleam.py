import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

# import pytorch_gleam.callbacks
# import pytorch_gleam.data.datasets


def main():
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
