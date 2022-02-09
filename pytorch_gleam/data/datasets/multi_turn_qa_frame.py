import json
from typing import List, Dict, Any, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import SequenceToSequenceBatchCollator
from pytorch_gleam.qa import QAModule


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


class MultiTurnQAFrameDataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        frame_path: Union[str, List[str]],
        label_name: str,
        qa: QAModule,
        label_map: Dict[str, int],
    ):
        super().__init__()
        self.frame_path = frame_path
        self.qa = qa
        self.label_name = label_name
        self.label_map = label_map

        self.examples = []
        if isinstance(self.frame_path, str):
            with open(self.frame_path) as f:
                self.frames = json.load(f)
        else:
            self.frames = {}
            for f_stage, f_path in enumerate(self.frame_path):
                with open(f_path) as f:
                    s_frames = json.load(f)
                for f_id, f in s_frames.items():
                    assert f_id not in self.frames, f"Duplicate frames: {f_id}"
                    self.frames[f_id] = f

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

    def read_path(self, data_path, stage=0):
        for ex in read_jsonl(data_path):
            ex_id = ex["id"]
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            for f_id, f_label in ex[self.label_name].items():
                frame = self.frames[f_id]
                frame_text = frame["text"]
                if f_label not in self.label_map:
                    continue

                ex_label = self.label_map[f_label]

                for q_id, input_ids, attention_mask, label_ids in self.qa.generate(
                    body=ex_text, label=f_label, context=frame_text
                ):
                    example = {
                        "ids": f"{ex_id}|{f_id}||{q_id}",
                        "label": ex_label,
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "label_ids": label_ids,
                    }

                    self.examples.append(example)

    def display_length_percentiles(self, key='input_ids'):
        lengths = [len(x[key]) for x in self.examples]
        print(f'mean={np.mean(lengths):.0f}')
        print(f'90%={np.percentile(lengths, 90):.0f}')
        print(f'95%={np.percentile(lengths, 95):.0f}')
        print(f'min={np.min(lengths):.0f}')
        print(f'max={np.max(lengths):.0f}')

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]

        return example

    def worker_init_fn(self, _):
        pass


class MultiTurnQAFrameDataModule(BaseDataModule):
    def __init__(
        self,
        label_name: str,
        label_map: Dict[str, int],
        qa: QAModule,
        frame_path: Union[str, List[str]],
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        test_path: Union[str, List[str]] = None,
        predict_path: Union[str, List[str]] = None,
        max_label_seq_len: int = 64,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.qa = qa
        self.label_map = label_map
        self.max_label_seq_len = max_label_seq_len

        self.label_name = label_name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.frame_path = frame_path

        if self.train_path is not None:
            self.train_dataset = MultiTurnQAFrameDataset(
                qa=self.qa,
                data_path=self.train_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
            )
        if self.val_path is not None:
            self.val_dataset = MultiTurnQAFrameDataset(
                qa=self.qa,
                data_path=self.val_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
            )
        if self.test_path is not None:
            self.test_dataset = MultiTurnQAFrameDataset(
                qa=self.qa,
                data_path=self.test_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
            )
        if self.predict_path is not None:
            self.predict_dataset = MultiTurnQAFrameDataset(
                qa=self.qa,
                data_path=self.predict_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
            )

    def create_collator(self):
        return SequenceToSequenceBatchCollator(
            max_seq_len=self.max_seq_len,
            max_label_seq_len=self.max_label_seq_len,
            use_tpus=self.use_tpus,
        )
