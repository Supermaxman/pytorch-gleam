from torch.optim import AdamW
from transformers import Adafactor

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForPreTraining


# noinspection PyAbstractClass
class BertPreTrainLanguageModel(BaseLanguageModelForPreTraining):
    def __init__(
        self,
        optimizer: str = "adamw",
        *args,
        **kwargs,
    ):
        self.optimizer = optimizer.lower()
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
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer}")
        return optimizer
