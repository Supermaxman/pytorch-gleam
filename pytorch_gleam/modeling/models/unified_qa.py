from typing import Optional

import numpy as np
import torch
from transformers import Adafactor
from transformers.optimization import get_adafactor_schedule, get_constant_schedule

from pytorch_gleam.modeling.models.base_models import BaseLanguageModelForSeq2SeqLM
from pytorch_gleam.qa import MultiQATaskModule


# noinspection PyAbstractClass
class UnifiedQAForConditionalGeneration(BaseLanguageModelForSeq2SeqLM):
    def __init__(
        self,
        qa_task: MultiQATaskModule,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qa_task = qa_task

    def eval_epoch_end(self, outputs, stage):
        loss = torch.stack([x["loss"] for x in outputs], dim=0).mean().cpu()
        self.log(f"{stage}_loss", loss)

        results, labels, preds, t_ids = self.eval_outputs(outputs, stage)
        for val_name, val in results.items():
            self.log(val_name, val)

    def eval_outputs(self, outputs, stage):
        results = {}

        ex_ids = self.flatten([x["ids"] for x in outputs])
        # [count]
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu()
        preds = torch.cat([x["preds"] for x in outputs], dim=0).cpu()

        # TODO this metric needs to be task-specific
        accuracy = labels.eq(preds).float().mean()
        results[f"{stage}_accuracy"] = accuracy

        task_metrics = self.qa_task.calculate_metrics(ex_ids, labels, preds)
        total_metrics = []
        for task, metric in task_metrics.items():
            p_metric = metric[0]
            results[f"{stage}_{task}_metric"] = p_metric
            total_metrics.append(p_metric)

        avg_metric = np.mean(total_metrics)
        results[f"{stage}_metric"] = avg_metric

        # f1, p, r, cls_f1, cls_p, cls_r, cls_indices = self.metric(labels, preds)
        #
        # results[f"{stage}_f1"] = f1
        # results[f"{stage}_p"] = p
        # results[f"{stage}_r"] = r
        # TODO support qa_task specific metrics here
        # for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
        #     label_name = self.inv_label_map[cls_index]
        #     results[f"{stage}_{label_name}_f1"] = c_f1
        #     results[f"{stage}_{label_name}_p"] = c_p
        #     results[f"{stage}_{label_name}_r"] = c_r

        return results, labels, preds, ex_ids

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        result = self.predict_step(batch, batch_idx, dataloader_idx)
        return result

    def forward(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        label_ids = batch["label_ids"]
        lm_out = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=label_ids)
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
        ).cpu()
        ex_ids, preds = self.qa_task(qa_ids=batch["ids"], qa_responses=batch_preds)
        # TODO clean this up, must be better way
        ex_label_map = {}
        for ex_id, t_label in zip(batch["ids"], batch["labels"].tolist()):
            # ds_path, ds_id = ex_id.split("||")
            ex_label_map[ex_id] = t_label
        labels = []
        for ex_id in ex_ids:
            ex_label = ex_label_map[ex_id]
            labels.append(ex_label)
        labels = torch.tensor(labels, dtype=torch.long)
        results = {
            # [bsize]
            "ids": ex_ids,
            "labels": labels,
            "preds": preds,
            "loss": batch_loss,
        }
        return results

    @staticmethod
    def flatten(multi_list):
        return [item for sub_list in multi_list for item in sub_list]

    def setup(self, stage: Optional[str] = None):
        pass

    def configure_optimizers(self):
        # total_devices = self.trainer.num_nodes * self.trainer.num_gpus
        # # https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        # # https://github.com/PyTorchLightning/pytorch-lightning/issues/10430
        # train_dataloader = (
        #     self.trainer._data_connector._train_dataloader_source.dataloader()
        # )
        # train_batches = len(train_dataloader) // total_devices
        # # need to figure out how many batches will actually have gradient updates
        # train_batches = train_batches // self.trainer.accumulate_grad_batches
        # self.train_steps = self.trainer.max_epochs * train_batches
        # self.warm_up_steps = 10000

        params = self.parameters()
        if self.learning_rate == 0.0:
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
            # scheduler = get_constant_schedule_with_warmup(
            #     optimizer,
            #     num_warmup_steps=self.warm_up_steps,
            # )
            scheduler = get_constant_schedule(optimizer)
        opt_dict = {
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
        return opt_dict
