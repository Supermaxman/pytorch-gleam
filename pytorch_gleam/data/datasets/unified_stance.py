from typing import Any, Dict, List, Union

import numpy as np
import torch
import ujson as json
from torch.utils.data import Dataset

from pytorch_gleam.data.collators import SequenceToSequenceBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
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

    def __init__(self, qa_task: MultiQATaskModule, split: str):
        super().__init__()
        self.qa_task = qa_task
        self.split = split

    def load(self, data_path):
        self.examples = self.qa_task.load(data_path, self.split)
        self.qa_task = None

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
        pickle_path: str = None,
        max_label_seq_len: int = 6,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qa_task = qa_task
        self.max_label_seq_len = max_label_seq_len
        self.pickle_path = pickle_path

        if self.train_path is not None:
            self.train_dataset = self.load_or_create(
                UnifiedQADataset,
                self.train_path,
                qa_task=self.qa_task,
                split="train",
                pickle_path=self.pickle_path,
            )
        if self.val_path is not None:
            self.val_dataset = self.load_or_create(
                UnifiedQADataset,
                self.val_path,
                qa_task=self.qa_task,
                split="val",
                pickle_path=self.pickle_path,
            )
        if self.test_path is not None:
            self.test_dataset = self.load_or_create(
                UnifiedQADataset,
                self.test_path,
                qa_task=self.qa_task,
                split="test",
                pickle_path=self.pickle_path,
            )
        if self.predict_path is not None:
            raise NotImplementedError()

    def create_collator(self):
        return SequenceToSequenceBatchCollator(
            max_seq_len=self.max_seq_len,
            max_label_seq_len=self.max_label_seq_len,
            use_tpus=self.use_tpus,
        )
