import math
from typing import Dict, Optional

import torch
from torch.optim.lr_scheduler import LambdaLR

from pytorch_gleam.modeling.layers.values import MultiValuesModule
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import ThresholdModule


# noinspection PyAbstractClass
class MultiClassFrameMultiValuesLanguageModel(BaseLanguageModel):
    def __init__(
        self,
        label_map: Dict[str, int],
        threshold: ThresholdModule,
        metric: Metric,
        values: MultiValuesModule,
        num_threshold_steps: int = 100,
        update_threshold: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.label_map = label_map
        self.values = values
        self.num_classes = len(label_map)
        self.threshold = threshold
        self.num_threshold_steps = num_threshold_steps
        self.update_threshold = update_threshold

        self.cls_layer = torch.nn.Linear(in_features=self.values.output_dim, out_features=self.num_classes)
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.f_dropout = torch.nn.Dropout(p=self.hidden_dropout_prob)
        self.metric = metric

        self.criterion = torch.nn.CrossEntropyLoss(reduction="none")
        self.score_func = torch.nn.Softmax(dim=-1)
        self.outputs = []

    def forward(self, batch, return_probs=False):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"]
        else:
            token_type_ids = None
        # [bsize, seq_len, hidden_size]
        contextualized_embeddings = self.lm_step(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        # use self.values
        values_outputs = self.values(
            contextualized_embeddings,
            embeddings_mask=attention_mask,
            cultural_mask=batch["cultural_mask"],
            moral_mask=batch["moral_mask"],
            post_mask=(token_type_ids == 1).long() if token_type_ids is not None else None,
            frame_mask=(token_type_ids == 0).long() if token_type_ids is not None else None,
        )
        output_features = values_outputs["output_features"]
        # [bsize, hidden_size]
        # lm_output = contextualized_embeddings[:, 0]
        # TODO add dropout
        # output_features = self.f_dropout(output_features)
        # [bsize, num_classes]
        logits = self.cls_layer(output_features)
        if return_probs:
            probs = {k: v for k, v in output_features.items() if "probs" in k}
            return logits, probs
        return logits

    def eval_epoch_end(self, outputs, stage):
        loss = torch.cat([x["loss"] for x in outputs], dim=0).mean().cpu()
        self.log(f"{stage}_loss", loss)
        self.threshold.cpu()

        results, labels, preds, t_ids = self.eval_outputs(
            outputs, stage, self.num_threshold_steps, self.update_threshold
        )
        for val_name, val in results.items():
            self.log(val_name, val)

        self.threshold.to(self.device)

    def eval_outputs(self, outputs, stage, num_threshold_steps=100, update_threshold=True):
        results = {}

        t_ids = self.flatten([x["ids"] for x in outputs])
        # [count]
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()
        # [count, num_classes]
        scores = torch.cat([x["scores"] for x in outputs], dim=0).cpu()

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
            label_name = self.inv_label_map[cls_index]
            results[f"{stage}_{label_name}_f1"] = c_f1
            results[f"{stage}_{label_name}_p"] = c_p
            results[f"{stage}_{label_name}_r"] = c_r
            results[f"{stage}_{cls_index}_f1"] = c_f1
            results[f"{stage}_{cls_index}_p"] = c_p
            results[f"{stage}_{cls_index}_r"] = c_r

        return results, labels, preds, t_ids

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        logits, probs = self(batch, return_probs=True)

        scores = self.score_func(logits)
        preds = self.threshold(scores)
        results = {
            # [bsize]
            "ids": batch["ids"],
            "labels": batch["labels"],
            "logits": logits,
            "scores": scores,
            "preds": preds,
        }
        if "labels" in batch:
            loss = self.loss(logits, batch["labels"])
            results["loss"] = loss
        for k, v in probs.items():
            results[k] = v
        return results

    def loss(self, logits, labels):
        loss = self.criterion(logits, labels)
        return loss

    def training_step(self, batch, batch_idx):
        batch_logits = self(batch)
        batch_labels = batch["labels"]
        batch_loss = self.loss(batch_logits, batch_labels)
        loss = batch_loss.mean()
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        results = self.eval_step(batch, batch_idx, dataloader_idx)
        return results

    def configure_optimizers(self):
        params = self._get_optimizer_params(self.weight_decay)
        optimizer = torch.optim.AdamW(
            params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = WarmupLR(
            optimizer,
            num_warmup_steps=int(math.ceil(self.lr_warm_up * self.trainer.estimated_stepping_batches)),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        optimizer_dict = {
            "optimizer": optimizer,
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

        return optimizer_dict

    def _get_optimizer_params(self, weight_decay):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_params

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "fit":
            self.update_threshold = True

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        batch_outputs = self.eval_step(batch, batch_idx, dataloader_idx)
        self.outputs.append(batch_outputs)

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        batch_outputs = self.eval_step(batch, batch_idx, dataloader_idx)
        self.outputs.append(batch_outputs)

    def on_validation_epoch_end(self):
        self.eval_epoch_end(self.outputs, "val")
        self.outputs.clear()

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.outputs, "test")
        self.outputs.clear()

    @staticmethod
    def flatten(multi_list):
        return [item for sub_list in multi_list for item in sub_list]


class WarmupLR(LambdaLR):
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
