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
        loss = torch.stack([x["loss"] for x in outputs], dim=0).mean().cpu()
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
        # TODO more metrics than just loss
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits
        loss = outputs.loss
        return loss

    def training_step(self, batch, batch_idx):
        batch_loss = self(batch)
        print(batch_loss.shape)
        exit()
        loss = batch_loss.mean()
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch_loss = self(batch)
        loss = batch_loss.mean()
        results = {
            "loss": loss,
        }
        return results
