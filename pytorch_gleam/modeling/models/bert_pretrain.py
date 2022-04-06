import torch
import torch.nn as nn

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForPreTraining


# noinspection PyAbstractClass
class BertPreTrainLanguageModel(BaseLanguageModelForPreTraining):
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loss_func = nn.CrossEntropyLoss(reduction="none")

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
            # labels=batch["masked_lm_labels"],
            # next_sentence_label=batch["next_sentence_labels"],
        )
        labels = batch["masked_lm_labels"]
        next_sentence_label = batch["next_sentence_labels"]

        # TODO more metrics than just loss
        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits
        masked_lm_loss = self.loss_func(prediction_logits.view(-1, self.lm.config.vocab_size), labels.view(-1))
        next_sentence_loss = self.loss_func(seq_relationship_logits.view(-1, 2), next_sentence_label.view(-1))
        masked_lm_loss = masked_lm_loss.mean()
        next_sentence_loss = next_sentence_loss.mean()
        total_loss = masked_lm_loss + next_sentence_loss

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
        params = self._get_optimizer_params(self.weight_decay)
        optimizer = torch.optim.AdamW(params, lr=self.learning_rate, weight_decay=self.weight_decay)
        # optimizer = AdamW(
        #     params,
        #     lr=self.learning_rate,
        #     weight_decay=self.weight_decay,
        #     correct_bias=False,
        # )
        # scheduler = get_constant_schedule(optimizer)
        # opt_dict = {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         # REQUIRED: The scheduler instance
        #         "scheduler": scheduler,
        #         # The unit of the scheduler's step size, could also be 'step'.
        #         # 'epoch' updates the scheduler on epoch end whereas 'step'
        #         # updates it after a optimizer update.
        #         "interval": "step",
        #         # How many epochs/steps should pass between calls to
        #         # `scheduler.step()`. 1 corresponds to updating the learning
        #         # rate after every epoch/step.
        #         "frequency": 1,
        #     },
        # }
        # return opt_dict
        return optimizer
