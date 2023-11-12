import random
from typing import Any, Dict, List, Union

import torch
import ujson as json
from torch.utils.data import Dataset

from pytorch_gleam.data.collators import MultiClassFrameValuesBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig


class MultiClassFrameValuesDataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        frame_path: Union[str, List[str]],
        label_name: str,
        label_map: Dict[str, int],
        value_list: List[str],
        preprocess_config: TweetPreprocessConfig,
        skip_unknown_labels: bool = False,
        all_frames: bool = False,
        gold_ratio: float = None,
    ):
        super().__init__()
        self.frame_path = frame_path
        self.label_name = label_name
        self.label_map = label_map
        self.value_list = value_list
        self.preprocess_config = preprocess_config
        self.skip_unknown_labels = skip_unknown_labels
        self.all_frames = all_frames
        self.gold_ratio = gold_ratio
        self.values_map = {v: i for i, v in enumerate(self.value_list)}

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

        for f_id, frame in self.frames.items():
            frame["text"] = preprocess_tweet(frame["text"], self.preprocess_config)
            frame["value_ids"] = []
            for val, _ in frame["values"].items():
                if val in self.values_map:
                    frame["value_ids"].append(self.values_map[val])
                elif self.skip_unknown_labels:
                    continue
                else:
                    raise ValueError(f"Unknown value: {val}")

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

        if self.gold_ratio is not None:
            self.examples = self.gold_sample(self.examples)

    def gold_sample(self, examples):
        gold_examples = [ex for ex in examples if ex["gold"]]
        non_gold_examples = [ex for ex in examples if not ex["gold"]]
        num_gold = len(gold_examples)
        num_non_gold = len(non_gold_examples)
        total = num_gold + num_non_gold
        assert num_gold > 0, "No gold examples found"
        assert num_non_gold > 0, "No non-gold examples found"
        non_gold_ratio = 1.0 - self.gold_ratio
        gold_ratio = num_gold / total
        if gold_ratio < self.gold_ratio:
            random.shuffle(non_gold_examples)
            non_gold_examples = non_gold_examples[: int(non_gold_ratio * num_gold)]
        return gold_examples + non_gold_examples

    def read_path(self, data_path, stage=0):
        for ex in read_jsonl(data_path):
            ex_id = ex["id"]
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
            ex_text = preprocess_tweet(ex_text, self.preprocess_config)

            if self.all_frames:
                for f_id, frame in self.frames.items():
                    frame_text = frame["text"]
                    ex_label = 0
                    gold = False
                    if f_id in ex[self.label_name]:
                        gold = True
                        f_label = ex[self.label_name][f_id]
                        f_label = f_label.replace(" ", "_")
                        if f_label in self.label_map:
                            ex_label = self.label_map[f_label]
                        elif self.skip_unknown_labels:
                            continue
                    example = {
                        "ids": f"{ex_id}|{f_id}",
                        "label": ex_label,
                        "text": (frame_text, ex_text),
                        "value_ids": frame["value_ids"],
                        "gold": gold,
                    }
                    self.examples.append(example)
            else:
                for f_id, f_label in ex[self.label_name].items():
                    frame = self.frames[f_id]
                    frame_text = frame["text"]
                    ex_label = 0
                    f_label = f_label.replace(" ", "_")
                    if f_label in self.label_map:
                        ex_label = self.label_map[f_label]
                    elif self.skip_unknown_labels:
                        continue
                    example = {
                        "ids": f"{ex_id}|{f_id}",
                        "label": ex_label,
                        "text": (frame_text, ex_text),
                        "value_ids": frame["value_ids"],
                        "gold": True,
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


class MultiClassFrameValuesDataModule(BaseDataModule):
    def __init__(
        self,
        label_name: str,
        label_map: Dict[str, int],
        frame_path: Union[str, List[str]],
        value_list: List[str],
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        test_path: Union[str, List[str]] = None,
        predict_path: Union[str, List[str]] = None,
        preprocess_config: TweetPreprocessConfig = None,
        skip_unknown_labels: bool = False,
        all_frames: bool = False,
        gold_ratio: float = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if preprocess_config is None:
            preprocess_config = TweetPreprocessConfig()
        self.preprocess_config = preprocess_config
        self.label_map = label_map
        self.all_frames = all_frames

        self.label_name = label_name
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.frame_path = frame_path
        self.skip_unknown_labels = skip_unknown_labels
        self.gold_ratio = gold_ratio
        self.value_list = value_list

        if self.train_path is not None:
            self.train_dataset = MultiClassFrameValuesDataset(
                data_path=self.train_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
                value_list=self.value_list,
                preprocess_config=preprocess_config,
                skip_unknown_labels=self.skip_unknown_labels,
                all_frames=self.all_frames,
                gold_ratio=self.gold_ratio,
            )
        if self.val_path is not None:
            self.val_dataset = MultiClassFrameValuesDataset(
                data_path=self.val_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
                value_list=self.value_list,
                preprocess_config=preprocess_config,
                skip_unknown_labels=self.skip_unknown_labels,
                all_frames=self.all_frames,
                gold_ratio=self.gold_ratio,
            )
        if self.test_path is not None:
            self.test_dataset = MultiClassFrameValuesDataset(
                data_path=self.test_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
                value_list=self.value_list,
                preprocess_config=preprocess_config,
                skip_unknown_labels=self.skip_unknown_labels,
                all_frames=self.all_frames,
                gold_ratio=self.gold_ratio,
            )
        if self.predict_path is not None:
            self.predict_dataset = MultiClassFrameValuesDataset(
                data_path=self.predict_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                label_map=self.label_map,
                value_list=self.value_list,
                preprocess_config=preprocess_config,
                skip_unknown_labels=self.skip_unknown_labels,
                all_frames=self.all_frames,
            )

    def create_collator(self):
        return MultiClassFrameValuesBatchCollator(
            self.tokenizer,
            self.value_list,
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )
