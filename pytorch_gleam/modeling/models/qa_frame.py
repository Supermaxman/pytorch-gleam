from typing import Dict

import torch
from transformers import Adafactor, get_constant_schedule_with_warmup
from transformers.optimization import get_adafactor_schedule

from pytorch_gleam.qa import QAModule
from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForSeq2SeqLM
from pytorch_gleam.modeling.metrics import Metric
import torch.nn.functional as F


# noinspection PyAbstractClass
class MultiTurnQAForConditionalGeneration(BaseLanguageModelForSeq2SeqLM):
    def __init__(
        self,
        label_map: Dict[str, int],
        qa: QAModule,
        metric: Metric,
        *args,
        **kwargs,
    ):
        r"""
        Multi-Class Language Model for baseline n-way classification tasks.

        Args:

                label_map: Dictionary mapping from name of class to class idx, used to determine
                        size of final softmax layer along with class-specific metrics. Class with zero idx is
                        considered the negative class.

                qa: QA module to use for system predictions.

                metric: Metric to evaluate overall performance. Typically Macro or Micro F1.

                pre_model_name: Name of pre-trained model from huggingface. See https://huggingface.co/

                pre_model_type: Type of pre-trained model.
                        Default: [`AutoModel`].

                learning_rate: Maximum learning rate. Learning rate will warm up from ``0`` to ``learning_rate`` over
                        ``lr_warm_up`` training steps, and will then decay from ``learning_rate`` to ``0`` linearly over
                         the remaining ``1.0-lr_warm_up`` training steps.

                weight_decay: How much weight decay to apply in the AdamW optimizer.
                        Default: ``0.0``.

                lr_warm_up: The percent of training steps to warm up learning rate from ``0`` to ``learning_rate``.
                        Default: ``0.1``.

                load_pre_model: If ``False``, Model structure will load from pre_model_name, but weights will not be
                        initialized. Cuts down on model load time if you plan on loading your model from a checkpoint,
                        as there is no reason to initialize your model twice.
                        Default: ``True``.

                torch_cache_dir: If provided, cache directory for loading models. Defaults to huggingface default.
                        Default: ``None``.

        """

        super().__init__(*args, **kwargs)
        self.label_map = label_map
        self.num_classes = len(label_map)
        self.qa = qa

        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.metric = metric

    def eval_epoch_end(self, outputs, stage):
        loss = torch.stack([x["loss"] for x in outputs], dim=0).mean().cpu()
        self.log(f"{stage}_loss", loss)

        results, labels, preds, t_ids = self.eval_outputs(outputs, stage)
        for val_name, val in results.items():
            self.log(val_name, val)

    def eval_outputs(self, outputs, stage):
        results = {}

        tq_ids = self.flatten([x["ids"] for x in outputs])
        # [count]
        tq_labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()
        ex_label_map = {}
        for t_id, t_label in zip(tq_ids, tq_labels.tolist()):
            ex_id, q_id = t_id.split("||")
            ex_label_map[ex_id] = t_label

        # [count, max_seq_len]
        # need to pad to max length
        max_pred_length = max([torch.max(x["pred_ids"]) for x in outputs]).item()
        pred_ids = torch.cat(
            [
                F.pad(x["pred_ids"], (0, max_pred_length - x["pred_ids"].shape[1]))
                for x in outputs
            ],
            dim=0,
        ).cpu()
        ex_ids, preds = self.qa(qa_ids=tq_ids, qa_responses=pred_ids)
        labels = []
        for ex_id in ex_ids:
            ex_label = ex_label_map[ex_id]
            labels.append(ex_label)
        labels = torch.tensor(labels, dtype=torch.long)

        f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(labels, preds)

        results[f"{stage}_f1"] = f1
        results[f"{stage}_p"] = p
        results[f"{stage}_r"] = r

        for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
            label_name = self.inv_label_map[cls_index]
            results[f"{stage}_{label_name}_f1"] = c_f1
            results[f"{stage}_{label_name}_p"] = c_p
            results[f"{stage}_{label_name}_r"] = c_r

        return results, labels, preds, ex_ids

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        result = self.predict_step(batch, batch_idx, dataloader_idx)
        return result

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label_ids = batch["label_ids"]
        lm_out = self.lm(
            input_ids=input_ids, attention_mask=attention_mask, labels=label_ids
        )
        loss = lm_out.loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self(batch)
        self.log("train_loss", loss)
        result = {"loss": loss}
        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        batch_loss = self(batch)
        # noinspection PyUnresolvedReferences
        batch_preds = self.lm.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            # TODO add generator args to model
            # **generator_args,
        )

        results = {
            # [bsize]
            "ids": batch["ids"],
            "labels": batch["labels"],
            "pred_ids": batch_preds,
            "loss": batch_loss,
        }
        return results

    @staticmethod
    def flatten(multi_list):
        return [item for sub_list in multi_list for item in sub_list]

    def configure_optimizers(self):
        params = self.parameters()
        if self.learning_rate is None:
            optimizer = Adafactor(
                params,
                scale_parameter=True,
                relative_step=True,
                warmup_init=True,
                lr=None,
            )
            scheduler = get_adafactor_schedule(optimizer)
        else:
            optimizer = Adafactor(
                params,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=self.learning_rate,
            )
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.lr_warm_up * self.train_steps,
            )
        return [optimizer], [scheduler]
