import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForSeq2SeqLM

from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.modeling.models.base_models import BasePreModel
from pytorch_gleam.modeling.thresholds import ThresholdModule


# noinspection PyAbstractClass
class DirectStanceLanguageModel(BasePreModel):
    def __init__(
        self,
        pre_model_name: str,
        threshold: ThresholdModule,
        metric: Metric,
        num_classes: int = 3,
        num_val_seeds: int = 1,
        num_threshold_steps: int = 100,
        update_threshold: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(pre_model_name, AutoModelForSeq2SeqLM, *args, **kwargs)
        self.num_classes = num_classes
        self.num_val_seeds = num_val_seeds
        self.threshold = threshold
        self.num_threshold_steps = num_threshold_steps
        self.update_threshold = update_threshold
        self.metric = metric
        self.lm_loss = CrossEntropyLoss(ignore_index=-100, reduction="none")

    def lm_step(self, *args, **kwargs):
        return self.lm(*args, **kwargs)

    def forward(self, batch):
        # [bsize, seq_len]
        input_ids = batch["input_ids"]
        # [bsize, seq_len]
        attention_mask = batch["attention_mask"]
        # [bsize, target_seq_len]
        target_ids = batch["target_ids"]
        _, pad_target_seq_len = target_ids.shape

        # get logits here
        results = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        # [bsize, target_seq_len, vocab_size]
        logits = results.logits
        # [bsize * target_seq_len]
        loss = self.lm_loss(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        # [bsize, target_seq_len]
        loss = loss.view(-1, pad_target_seq_len)
        # [bsize]
        seq_lens = (target_ids != -100).float().sum(dim=-1)
        # [bsize]
        loss = loss.sum(dim=-1) / (seq_lens + 1e-8)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        loss = loss.mean()
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def eval_epoch_end(self, outputs, stage):
        scores = torch.cat([x["scores"] for x in outputs], dim=0).cpu().view(-1, self.num_classes)
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()[:: self.num_classes]
        labels_mask = F.one_hot(labels, num_classes=self.num_classes)
        loss = (-scores * labels_mask).mean()

        self.log(f"{stage}_loss", loss)
        self.threshold.cpu()

        results, labels, preds, t_ids = self.eval_outputs(
            outputs, stage, self.num_threshold_steps, self.update_threshold
        )
        for val_name, val in results.items():
            self.log(val_name, val)

        self.threshold.to(self.device)

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        result = self.predict_step(batch, batch_idx, dataloader_idx)
        return result

    def eval_outputs(self, outputs, stage, num_threshold_steps=100, update_threshold=True):
        results = {}

        # [count]
        t_ids = DirectStanceLanguageModel.flatten([x["ids"] for x in outputs])[:: self.num_classes]
        # [count]
        # m_ids = NoisyChannelLanguageModel.flatten([x["m_ids"] for x in outputs])[:: self.num_classes]
        # [count]
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()[:: self.num_classes]
        # [count, num_classes]
        scores = torch.cat([x["scores"] for x in outputs], dim=0).cpu().view(-1, self.num_classes)

        self.threshold.cpu()
        if update_threshold:
            m_min_score = torch.min(scores).item()
            m_max_score = torch.max(scores).item()
            # check 100 values between min and max
            if abs(m_min_score - m_max_score) < 1e-6:
                m_max_score += 10
            m_delta = (m_max_score - m_min_score) / num_threshold_steps
            max_threshold, max_metrics = self.metric.best(
                labels,
                scores,
                self.threshold,
                threshold_min=m_min_score,
                threshold_max=m_max_score,
                threshold_delta=m_delta,
            )
            self.threshold.update_thresholds(max_threshold)
        preds = self.threshold(scores)

        f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(labels, preds)

        results[f"{stage}_f1"] = f1
        results[f"{stage}_p"] = p
        results[f"{stage}_r"] = r

        for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
            results[f"{stage}_{cls_index}_f1"] = c_f1
            results[f"{stage}_{cls_index}_p"] = c_p
            results[f"{stage}_{cls_index}_r"] = c_r

        return results, labels, preds, t_ids

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self(batch)
        results = {
            "ids": batch["ids"],
            "m_ids": batch["m_ids"],
            "labels": batch["labels"],
            "m_label_idx": batch["m_label_idx"],
            "stages": batch["stages"],
            "scores": -loss,
        }
        return results

    @staticmethod
    def flatten(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]
