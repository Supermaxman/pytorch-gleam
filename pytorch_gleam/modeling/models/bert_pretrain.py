from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import Adafactor

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForPreTraining


# noinspection PyAbstractClass
class BertPreTrainLanguageModel(BaseLanguageModelForPreTraining):
    def __init__(
        self,
        optimizer: str = "adamw",
        warmup_steps: int = 10000,
        train_steps: int = 10000000,
        *args,
        **kwargs,
    ):
        self.optimizer = optimizer.lower()
        self.warmup_steps = warmup_steps
        self.train_steps = train_steps
        super().__init__(*args, **kwargs)

    def eval_epoch_end(self, outputs, stage):
        pass

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        result = self.predict_step(batch, batch_idx, dataloader_idx)
        self.log("val_loss", result["loss"])
        return result

    def forward(self, batch):
        outputs = self.lm(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_type_ids"],
            labels=batch["masked_lm_labels"],
            next_sentence_label=batch["next_sentence_labels"],
        )
        total_loss = outputs.loss
        return total_loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self(batch)
        results = {
            "loss": loss,
        }
        return results

    def configure_optimizers(self):
        params = self.parameters()
        if self.optimizer == "adamw":
            optimizer = AdamW(params, lr=self.learning_rate, weight_decay=0.0)
        elif self.optimizer == "adafactor":
            optimizer = Adafactor(
                params,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=self.learning_rate,
            )
        elif self.optimizer == "bert":
            params = self._get_optimizer_params(self.weight_decay)
            opt = AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay, betas=(0.9, 0.999), eps=1e-6)

            scheduler = BertLR(
                opt,
                num_warmup_steps=self.warmup_steps,
                # TODO use self.trainer.estimated_stepping_batches, but not stable yet
                num_training_steps=self.train_steps,
            )
            optimizer = {
                "optimizer": opt,
                "lr_scheduler": {
                    # REQUIRED: The scheduler instance
                    "scheduler": scheduler,
                    # The unit of the scheduler's step size, could also be 'step'.
                    # 'epoch' updates the scheduler on epoch end whereas 'step'
                    # updates it after a optimizer update.
                    "interval": "step",
                    # How many epochs/steps should pass between calls to
                    # `scheduler.step()`. 1 corresponds to updating the learning
                    # rate after every epoch/step.
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        return optimizer


class BertLR(LambdaLR):
    def __init__(self, optimizer, num_warmup_steps: int, num_training_steps: int, last_epoch=-1):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer=optimizer, lr_lambda=self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, current_step: int):
        if current_step < self.num_warmup_steps:
            return float(current_step) / float(max(1, self.num_warmup_steps))
        return max(
            0.0,
            float(self.num_training_steps - current_step)
            / float(max(1, self.num_training_steps - self.num_warmup_steps)),
        )
