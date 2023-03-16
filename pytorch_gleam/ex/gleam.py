import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI


def main():
    LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
    )


if __name__ == "__main__":
    main()
