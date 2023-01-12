from typing import Any, Dict, List, Union

import torch
import ujson as json
from torch.utils.data import Dataset

from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig


class MisinfoStanceDataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]
    preprocess_config: TweetPreprocessConfig

    def __init__(
        self,
        data_path: Union[str, List[str]],
        misinfo_path: str,
        preprocess_config: TweetPreprocessConfig,
        label_name: str = "misinfo",
    ):
        super().__init__()
        self.preprocess_config = preprocess_config
        self.label_name = label_name
        self.label_map = {
            "No Stance": 0,
            "no_stance": 0,
            "Accept": 1,
            "agree": 1,
            "Reject": 2,
            "disagree": 2,
        }
        with open(misinfo_path) as f:
            self.misinfo = json.load(f)

        self.examples = []
        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

    def read_path(self, data_path, stage=0):
        for ex in read_jsonl(data_path):
            ex_id = ex["id"]
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
            ex_text = preprocess_tweet(ex_text, self.preprocess_config)

            ex_labels = ex[self.label_name]
            for m_id, m_label in ex_labels.items():
                if m_id not in self.misinfo:
                    print(f"MISSING M_ID: {m_id}")
                    continue
                if m_label not in self.label_map:
                    # print(f"MISSING M_LABEL: {m_label}")
                    continue
                m_label_idx = self.label_map[m_label]
                m = self.misinfo[m_id]
                m_text = m["text"]
                example = {
                    "ex_id": ex_id,
                    "m_id": m_id,
                    "ex_text": ex_text,
                    "m_text": m_text,
                    "m_label": m_label_idx,
                    "stage": stage,
                }
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]

        return example

    def worker_init_fn(self, _):
        pass
