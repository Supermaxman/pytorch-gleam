from typing import Optional

import torch
import torchmetrics

from pytorch_gleam.modeling.losses import ContrastiveLoss
from pytorch_gleam.modeling.metrics import F1PRMultiClassMetric
from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule


# noinspection PyAbstractClass
class ContrastiveFrameLanguageModel(BaseLanguageModel):
    def __init__(
        self, loss: ContrastiveLoss, update_threshold: bool = False, num_threshold_steps: int = 100, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.score_layer = torch.nn.Linear(in_features=self.hidden_size, out_features=1)
        self.f_dropout = torch.nn.Dropout(p=self.hidden_dropout_prob)
        self.loss = loss
        self.num_threshold_steps = num_threshold_steps
        self.update_threshold = update_threshold
        self.metric = torchmetrics.Accuracy()
        self.threshold = ThresholdModule()
        self.rank_metric = F1PRMultiClassMetric(num_classes=2, mode="micro")

    def setup(self, stage: Optional[str] = None) -> None:
        super().setup(stage)
        if stage == "fit":
            self.update_threshold = True

    def eval_epoch_end(self, outputs, stage):
        acc = self.metric.compute()

        self.log(f"{stage}_accuracy", acc)

        self.metric.reset()

        loss = torch.cat([x["loss"] for x in outputs], dim=0).mean()
        self.log(f"{stage}_loss", loss)

        self.threshold.cpu()

        # t_ids = self.flatten([x["ids"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs], dim=0)
        scores = torch.cat([x["scores"] for x in outputs], dim=0)

        if self.update_threshold:
            m_min_score = torch.min(scores).item()
            m_max_score = torch.max(scores).item()
            # check 100 values between min and max
            if abs(m_min_score - m_max_score) < 1e-6:
                m_max_score += 10
            m_delta = (m_max_score - m_min_score) / self.num_threshold_steps
            max_threshold, max_metrics = self.rank_metric.best(
                labels,
                scores,
                self.threshold,
                threshold_min=m_min_score,
                threshold_max=m_max_score,
                threshold_delta=m_delta,
            )
            self.threshold.update_thresholds(max_threshold)
        preds = self.threshold(scores)

        f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.rank_metric(labels, preds)

        self.log(f"{stage}_f1", f1)
        self.log(f"{stage}_p", p)
        self.log(f"{stage}_r", r)

        self.threshold.to(self.device)

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        scores = self(batch)
        pos_samples = batch["pos_samples"]
        pos_scores = scores[:, :pos_samples]
        neg_scores = scores[:, pos_samples:]

        labels = torch.cat([torch.ones_like(pos_scores).long(), torch.zeros_like(neg_scores).long()], dim=0)
        correct = pos_scores.lt(neg_scores).long()
        self.metric(correct, torch.ones_like(correct).long())
        loss = self.loss(pos_scores, neg_scores)
        results = {
            "ids": batch["ids"],
            "labels": labels.cpu(),
            "scores": scores.cpu(),
            "loss": loss.cpu(),
        }
        return results

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

    @staticmethod
    def flatten(multi_list):
        return [item for sub_list in multi_list for item in sub_list]
