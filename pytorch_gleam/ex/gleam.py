import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI


def main():
    cli = LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        run=True,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()