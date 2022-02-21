import ujson as json
from typing import List, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import SequenceToSequenceBatchCollator
from pytorch_gleam.qa import MultiQATaskModule


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


class UnifiedQADataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]

    def __init__(self, qa_task: MultiQATaskModule, split: str, cache_path: str):
        super().__init__()
        self.qa_task = qa_task
        self.split = split
        self.cache_path = cache_path
        self.examples = qa_task.load(self.cache_path, self.split)
        self.display_length_percentiles()
        exit()

    def display_length_percentiles(self, key="input_ids"):
        lengths = [len(x[key]) for x in self.examples]
        print(f"mean={np.mean(lengths):.0f}")
        print(f"90%={np.percentile(lengths, 90):.0f}")
        print(f"95%={np.percentile(lengths, 95):.0f}")
        print(f"min={np.min(lengths):.0f}")
        print(f"max={np.max(lengths):.0f}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]

        return example

    def worker_init_fn(self, _):
        pass


class UnifiedQADataModule(BaseDataModule):
    def __init__(
        self,
        qa_task: MultiQATaskModule,
        max_label_seq_len: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qa_task = qa_task
        self.max_label_seq_len = max_label_seq_len

        if self.train_path is not None:
            self.train_dataset = UnifiedQADataset(
                qa_task=self.qa_task, split="train", cache_path=self.train_path
            )
        if self.val_path is not None:
            self.val_dataset = UnifiedQADataset(
                qa_task=self.qa_task, split="val", cache_path=self.val_path
            )
        if self.test_path is not None:
            self.test_dataset = UnifiedQADataset(
                qa_task=self.qa_task, split="test", cache_path=self.test_path
            )
        if self.predict_path is not None:
            raise NotImplementedError()

    def create_collator(self):
        return SequenceToSequenceBatchCollator(
            max_seq_len=self.max_seq_len,
            max_label_seq_len=self.max_label_seq_len,
            use_tpus=self.use_tpus,
        )