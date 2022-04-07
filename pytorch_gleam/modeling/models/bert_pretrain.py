import torch

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForPreTraining


# noinspection PyAbstractClass
class BertPreTrainLanguageModel(BaseLanguageModelForPreTraining):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def eval_epoch_end(self, outputs, stage):
        loss = torch.stack([x["loss"] for x in outputs], dim=0).mean()
        self.log(f"{stage}_loss", loss)

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        result = self.predict_step(batch, batch_idx, dataloader_idx)
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
            "loss": loss.detach(),
        }
        return results

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.0)
        return optimizer
