import torch
import torchmetrics

from pytorch_gleam.modeling.losses import ContrastiveLoss
from pytorch_gleam.modeling.models.base_models import BaseLanguageModel


# noinspection PyAbstractClass
class ContrastiveFrameLanguageModel(BaseLanguageModel):
    def __init__(self, loss: ContrastiveLoss, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.score_layer = torch.nn.Linear(in_features=self.hidden_size, out_features=1)
        self.f_dropout = torch.nn.Dropout(p=self.hidden_dropout_prob)
        self.loss = loss
        self.metric = torchmetrics.Accuracy()

    def eval_epoch_end(self, outputs, stage):
        acc = self.metric.compute()

        self.log("val_accuracy", acc)

        self.metric.reset()

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        scores = self(batch)
        pos_samples = batch["pos_samples"]
        pos_scores = scores[:, :pos_samples]
        neg_scores = scores[:, pos_samples:]

        correct = pos_scores.lt(neg_scores).float()
        self.metric(correct, torch.ones_like(correct))
        loss = self.loss(pos_scores, neg_scores)
        self.log("val_loss", loss.mean())

    def training_step(self, batch, batch_idx):
        scores = self(batch)

        pos_samples = batch["pos_samples"]
        pos_scores = scores[:, :pos_samples]
        neg_scores = scores[:, pos_samples:]

        acc = pos_scores.lt(neg_scores).float().mean()
        loss = self.loss(pos_scores, neg_scores)
        loss = loss.mean()
        self.log("train_loss", loss)
        self.log("train_accuracy", acc)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        scores = self(batch)

        results = {
            "ids": batch["ids"],
            "p_ids": batch["p_ids"],
            "scores": scores,
        }
        return results

    def forward(self, batch):
        batch_size, num_sequences, pad_seq_len = batch["input_ids"].shape

        input_ids = batch["input_ids"].view(batch_size * num_sequences, pad_seq_len)
        attention_mask = batch["attention_mask"].view(batch_size * num_sequences, pad_seq_len)
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"].view(batch_size * num_sequences, pad_seq_len)
        else:
            token_type_ids = None
        # [batch_size * num_sequences, seq_len, hidden_size]
        contextualized_embeddings = self.lm_step(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # [batch_size * num_sequences, hidden_size]
        lm_output = contextualized_embeddings[:, 0]
        lm_output = self.f_dropout(lm_output)
        scores = self.score_layer(lm_output)
        scores = scores.view(batch_size, num_sequences, scores.shape[-1])
        return scores
