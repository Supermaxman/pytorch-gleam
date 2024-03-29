from collections import defaultdict
from typing import Optional

import numpy as np
import torch

from pytorch_gleam.inference import ConsistencyScoring
from pytorch_gleam.modeling.metrics import Metric
from pytorch_gleam.modeling.models.base_models import BaseLanguageModel
from pytorch_gleam.modeling.thresholds import MultiClassThresholdModule, ThresholdModule


# noinspection PyAbstractClass
class ContrastiveFrameStanceLanguageModel(BaseLanguageModel):
    def __init__(
        self,
        margin: float,
        infer: ConsistencyScoring,
        threshold: ThresholdModule,
        metric: Metric,
        m_metric: Metric,
        num_relations: int = 2,
        num_classes: int = 3,
        num_val_seeds: int = 1,
        num_threshold_steps: int = 100,
        update_threshold: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_relations = num_relations
        self.num_classes = num_classes
        self.margin = margin
        self.infer = infer
        self.num_val_seeds = num_val_seeds
        self.threshold = threshold
        self.m_metric = m_metric
        self.num_threshold_steps = num_threshold_steps
        self.update_threshold = update_threshold
        self.metric = metric
        self.lm_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.cls_layer = torch.nn.Linear(in_features=self.hidden_size, out_features=1)

    def setup(self, stage: Optional[str] = None):
        super().setup(stage)
        if stage == "fit":
            data_loader = self.trainer.datamodule.train_dataloader()
        elif stage == "test":
            data_loader = self.trainer.datamodule.test_dataloader()[0]
        elif stage == "val":
            data_loader = self.trainer.datamodule.val_dataloader()[0]
        elif stage == "predict":
            data_loader = self.trainer.datamodule.predict_dataloader()
        else:
            raise ValueError(f"Unknown stage: {stage}")
        misinfo = data_loader.dataset.misinfo
        for m_id, _ in misinfo.items():
            if m_id not in self.threshold:
                self.threshold[m_id] = MultiClassThresholdModule()

    def forward(self, batch):
        num_examples = batch["num_examples"]
        num_sequences_per_example = batch["num_sequences_per_example"]
        # [bsize * num_seq, seq_len]
        input_ids = batch["input_ids"]
        # [bsize * num_seq, seq_len]
        attention_mask = batch["attention_mask"]
        # [bsize * num_seq, seq_len]
        if "token_type_ids" in batch:
            token_type_ids = batch["token_type_ids"]
        else:
            token_type_ids = None
        # [bsize * num_seq, seq_len, hidden_size]
        contextualized_embeddings = self.lm_step(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )
        lm_output = contextualized_embeddings[:, 0]
        # [bsize * num_seq]
        logits = torch.unsqueeze(self.cls_layer(lm_output), -1)

        # [bsize * num_seq]
        loss = self.lm_loss(logits, torch.ones_like(logits))
        # [bsize, num_seq]
        loss = loss.view(num_examples, num_sequences_per_example)

        return loss

    @staticmethod
    def split_energy(loss, batch):
        pos_samples = batch["pos_samples"]
        pos_energy = loss[:, :pos_samples]
        # neg seq lens equal to pos seq lens since direction is same for both pos and neg
        neg_energy = loss[:, pos_samples:]
        return pos_energy, neg_energy

    def triplet_step(self, batch):
        loss = self(batch)
        pos_energy, neg_energy = self.split_energy(loss, batch)
        if self.margin != 0:
            loss = torch.relu(pos_energy - neg_energy + self.margin)
        else:
            loss = pos_energy - neg_energy
        accuracy = (pos_energy < neg_energy).float().mean(dim=-1)
        return loss, accuracy, pos_energy, neg_energy

    def training_step(self, batch, batch_idx):
        loss, accuracy, pos_energy, neg_energy = self.triplet_step(batch)
        accuracy = accuracy.mean()
        loss = loss.mean()
        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        self.log("train_pos_energy", pos_energy.mean())
        self.log("train_neg_energy", neg_energy.mean())
        train_energy_margin = ((neg_energy - pos_energy)).mean()
        self.log("train_energy_margin", train_energy_margin)
        # estimate how close we are getting to the prob ratio we want to achieve
        # average before exp to avoid outliers killing the estimate
        result = {"loss": loss}
        return result

    @staticmethod
    def infer_m_scores(infer, adj_list, stage_labels, stage, num_val_seeds=1):
        # always use stage 0 (val) for seeds
        seed_labels = stage_labels[0]
        seed_examples = [(ex_id, label) for (ex_id, label) in seed_labels.items() if label != 0]
        # if the stage is val then we have no test set, so pick
        # some number of seed examples from val and test on remaining val
        if stage == "val":
            seed_examples = seed_examples[:num_val_seeds]
            # make sure adj list only has val labeled data
            adj_list = [
                (u_id, v_id, uv_scores)
                for (u_id, v_id, uv_scores) in adj_list
                if u_id in seed_labels and v_id in seed_labels
            ]
        else:
            if len(seed_examples) == 0:
                seed_examples = [(ex_id, label) for (ex_id, label) in stage_labels[1].items() if label != 0][
                    :num_val_seeds
                ]
        seed_examples = {ex_id: label for (ex_id, label) in seed_examples}
        if len(adj_list) == 0 or len(seed_examples) == 0:
            node_scores = np.zeros([len(seed_labels), 3], dtype=np.float32)
            node_idx_map = {node: idx for (idx, node) in enumerate(seed_labels)}
        else:
            node_scores, node_idx_map = infer(adj_list, seed_examples)
        if stage == "test":
            eval_labels = stage_labels[1]
        else:
            eval_labels = stage_labels[0]
        scores = []
        # make sure we pack example scores in proper order
        for ex_id in eval_labels:
            # only happens if we have no seed examples
            if ex_id not in node_idx_map:
                scores.append(torch.zeros([3], dtype=torch.float32))
                continue
            ex_idx = node_idx_map[ex_id]
            ex_scores = torch.tensor(node_scores[ex_idx], dtype=torch.float32)
            scores.append(ex_scores)
        scores = torch.stack(scores, dim=0)
        return scores

    def eval_epoch_end(self, outputs, stage):
        triplet_eval_outputs, infer_eval_outputs = outputs
        triplet_eval_results = self.eval_triplet(triplet_eval_outputs, stage)
        for val_name, val in triplet_eval_results.items():
            self.log(val_name, val)

        self.threshold.cpu()

        infer_eval_results, labels, preds, t_ids, m_ids = self.eval_infer(
            infer_eval_outputs,
            stage,
            self.infer,
            self.threshold,
            self.m_metric,
            self.metric,
            self.num_threshold_steps,
            self.update_threshold,
            self.num_val_seeds,
        )
        for val_name, val in infer_eval_results.items():
            self.log(val_name, val)

        self.threshold.to(self.device)

    @staticmethod
    def eval_infer(
        infer_eval_outputs,
        stage,
        infer,
        threshold,
        m_metric,
        metric,
        num_threshold_steps=100,
        update_threshold=True,
        num_val_seeds=1,
    ):
        results = {}
        # stage 0 is validation
        # stage 1 is test
        m_adj_lists, m_stage_labels = ContrastiveFrameStanceLanguageModel.build_adj_list(infer_eval_outputs)

        for m_id in m_stage_labels:
            if m_id not in threshold:
                threshold[m_id] = MultiClassThresholdModule()

        m_s_ids = []
        m_s_m_ids = []
        m_s_labels = []
        m_s_preds = []
        for m_id, stage_labels in m_stage_labels.items():
            m_adj_list = m_adj_lists[m_id]
            m_threshold = threshold[m_id]
            if len(m_adj_list) == 0:
                continue
            m_ex_ids = []
            m_ex_m_ids = []
            m_ex_labels = []
            if stage != "val":
                eval_labels = stage_labels[1]
            else:
                eval_labels = stage_labels[0]
            for ex_id, label in eval_labels.items():
                m_ex_labels.append(label)
                m_ex_ids.append(ex_id)
                m_ex_m_ids.append(m_id)
            m_ex_labels = torch.tensor(m_ex_labels, dtype=torch.long)
            m_ex_scores = ContrastiveFrameStanceLanguageModel.infer_m_scores(
                infer, m_adj_list, stage_labels, stage, num_val_seeds
            )
            if update_threshold:
                m_min_score = torch.min(m_ex_scores).item()
                m_max_score = torch.max(m_ex_scores).item()
                # check 100 values between min and max
                if m_min_score == m_max_score:
                    m_max_score += 1
                m_delta = (m_max_score - m_min_score) / num_threshold_steps
                max_threshold, max_metrics = m_metric.best(
                    m_ex_labels,
                    m_ex_scores,
                    m_threshold,
                    threshold_min=m_min_score,
                    threshold_max=m_max_score,
                    threshold_delta=m_delta,
                )
                m_threshold.update_thresholds(max_threshold)

            m_ex_preds = m_threshold(m_ex_scores)
            m_f1, m_p, m_r, m_cls_f1, m_cls_p, m_cls_r, m_cls_indices = m_metric(m_ex_labels, m_ex_preds)
            results[f"{stage}_{m_id}_micro_f1"] = m_f1
            results[f"{stage}_{m_id}_micro_p"] = m_p
            results[f"{stage}_{m_id}_micro_r"] = m_r
            results[f"{stage}_{m_id}_threshold"] = m_threshold.thresholds.item()
            for cls_index, c_f1, c_p, c_r in zip(m_cls_indices, m_cls_f1, m_cls_p, m_cls_r):
                results[f"{stage}_{m_id}_{cls_index}_f1"] = c_f1
                results[f"{stage}_{m_id}_{cls_index}_p"] = c_p
                results[f"{stage}_{m_id}_{cls_index}_r"] = c_r
            m_s_ids.extend(m_ex_ids)
            m_s_m_ids.extend(m_ex_m_ids)
            m_s_labels.append(m_ex_labels)
            m_s_preds.append(m_ex_preds)

        m_s_labels = torch.cat(m_s_labels, dim=0)
        m_s_preds = torch.cat(m_s_preds, dim=0)
        f1, p, r, cls_f1, cls_p, cls_r, cls_indices = metric(m_s_labels, m_s_preds)
        micro_f1, micro_p, micro_r, _, _, _, _ = m_metric(m_s_labels, m_s_preds)
        results[f"{stage}_f1"] = f1
        results[f"{stage}_p"] = p
        results[f"{stage}_r"] = r
        results[f"{stage}_micro_f1"] = micro_f1
        results[f"{stage}_micro_p"] = micro_p
        results[f"{stage}_micro_r"] = micro_r
        for cls_index, c_f1, c_p, c_r in zip(cls_indices, cls_f1, cls_p, cls_r):
            results[f"{stage}_{cls_index}_f1"] = c_f1
            results[f"{stage}_{cls_index}_p"] = c_p
            results[f"{stage}_{cls_index}_r"] = c_r

        return results, m_s_labels, m_s_preds, m_s_ids, m_s_m_ids

    @staticmethod
    def eval_triplet(triplet_eval_outputs, stage):
        loss = torch.cat([x["loss"] for x in triplet_eval_outputs], dim=0).mean()
        accuracy = torch.cat([x["accuracy"] for x in triplet_eval_outputs], dim=0).mean()
        results = {f"{stage}_loss": loss, f"{stage}_accuracy": accuracy}
        return results

    def eval_step(self, batch, batch_idx, dataloader_idx=None):
        if dataloader_idx is None or dataloader_idx == 0:
            loss, accuracy, _, _ = self.triplet_step(batch)
            result = {
                "loss": loss,
                "accuracy": accuracy,
            }
        else:
            result = self.predict_step(batch, batch_idx, dataloader_idx)

        return result

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        energies = self(batch)
        num_examples = batch["num_examples"]
        num_sequences_per_example = batch["num_sequences_per_example"]
        results = {
            # [bsize]
            "ids": batch["ids"],
            # [bsize]
            "m_ids": batch["m_ids"],
            # [bsize, num_pairs]
            "s_ids": batch["s_ids"],
            # [bsize]
            "s_ids_reshaped": batch["s_ids"][::2],
            # [bsize, num_pairs+1]
            "labels": batch["labels"],
            "labels_reshaped": batch["labels"].view(num_examples, num_sequences_per_example + 1),
            # [bsize, num_pairs+1]
            "stages": batch["stages"],
            "stages_reshaped": batch["stages"].view(num_examples, num_sequences_per_example + 1),
            # [bsize, num_pairs]
            "energies": energies,
            "input_ids": batch["input_ids"].view(num_examples, num_sequences_per_example, -1),
        }
        return results

    @staticmethod
    def flatten(list_of_lists):
        return [item for sublist in list_of_lists for item in sublist]

    @staticmethod
    def build_adj_list(outputs):
        # [count]
        t_ids = ContrastiveFrameStanceLanguageModel.flatten([x["ids"] for x in outputs])
        # [count]
        m_ids = ContrastiveFrameStanceLanguageModel.flatten([x["m_ids"] for x in outputs])
        # [count]
        # skip every other s_id since they are duplicates from two relationships
        p_ids = ContrastiveFrameStanceLanguageModel.flatten([x["s_ids"] for x in outputs])[::2]
        assert len(p_ids) == len(t_ids)
        assert len(m_ids) == len(t_ids)
        # [count, 3]
        # TODO don't hardcode this, but for now there's only two examples in each relationship
        labels = torch.cat([x["labels"] for x in outputs], dim=0).cpu().view(-1, 3)
        # [count, 3]
        stages = torch.cat([x["stages"] for x in outputs], dim=0).cpu().view(-1, 3)
        # [count]
        t_label = labels[:, 0]
        # [count]
        t_stage = stages[:, 0]
        assert len(t_label) == len(t_ids)
        assert len(t_stage) == len(t_ids)
        # labels and stage are duplicated here for each relation
        # [count, 1]
        p_labels = labels[:, 1].unsqueeze(dim=-1)
        # [count, 1]
        p_stage = stages[:, 1].unsqueeze(dim=-1)

        # [count, 1, num_relations]
        t_energies = torch.cat([x["energies"] for x in outputs], dim=0).cpu().unsqueeze(dim=1)
        assert t_energies.shape[0] == len(t_ids)

        m_adj_list = defaultdict(list)
        m_labels = defaultdict(lambda: defaultdict(dict))
        for ex_idx in range(len(t_ids)):
            ex_t_id = t_ids[ex_idx]
            ex_m_id = m_ids[ex_idx]
            ex_p_ids = [p_ids[ex_idx]]
            ex_t_label = t_label[ex_idx]
            ex_t_stage = int(t_stage[ex_idx])
            ex_p_labels = p_labels[ex_idx]
            ex_p_stage = p_stage[ex_idx]
            ex_t_energies = t_energies[ex_idx]
            m_labels[ex_m_id][ex_t_stage][ex_t_id] = ex_t_label.item()
            for p_idx in range(len(ex_p_ids)):
                ex_p_id = ex_p_ids[p_idx]
                ex_p_label = ex_p_labels[p_idx]
                ex_p_stage = int(ex_p_stage[p_idx])
                ex_tmp_energy = ex_t_energies[p_idx]
                m_adj_list[ex_m_id].append((ex_t_id, ex_p_id, ex_tmp_energy))
                m_labels[ex_m_id][ex_p_stage][ex_p_id] = ex_p_label.item()

        return m_adj_list, m_labels
