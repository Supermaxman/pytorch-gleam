import pytorch_lightning as pl
import torch
from pytorch_lightning.cli import LightningCLI


def main():
    cli = LightningCLI(
        pl.LightningModule,
        pl.LightningDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        save_config_kwargs={"overwrite": True},
        run=False,
    )
    torch._dynamo.config.verbose = True
    cli.model = torch.compile(cli.model, dynamic=True)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
