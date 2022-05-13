import itertools
from collections import defaultdict
from typing import Any, Dict, List, Union

import torch
import ujson as json
from pytorch_lightning.utilities.cli import DATAMODULE_REGISTRY
from torch.utils.data import Dataset

from pytorch_gleam.data.collators import ContrastiveFrameBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig


class ContrastiveFrameDataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        frame_path: Union[str, List[str]],
        label_name: str,
        tokenizer,
        preprocess_config: TweetPreprocessConfig,
    ):
        super().__init__()
        self.frame_path = frame_path
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.preprocess_config = preprocess_config

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

        self.pos_examples = defaultdict(list)
        self.neg_examples = defaultdict(list)
        self.hard_negatives = defaultdict(list)
        self.soft_negatives = defaultdict(list)

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

    def read_path(self, data_path, stage=0):
        examples = []
        for ex in read_jsonl(data_path):
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
            ex_text = preprocess_tweet(ex_text, self.preprocess_config)
            ex["text"] = ex_text
            ex_frames = ex[self.label_name]
            examples.append(ex)
            ex_pos_f_ids = []
            for f_id, frame in self.frames.items():
                if f_id in ex_frames:
                    f_label = ex_frames[f_id]
                    if f_label != "Not Relevant":
                        ex_pos_f_ids.append(f_id)
                        self.pos_examples[f_id].append(ex)
                    else:
                        self.hard_negatives[f_id].append(ex)
                        self.neg_examples[f_id].append(ex)
                else:
                    self.soft_negatives[f_id].append(ex)
                    self.neg_examples[f_id].append(ex)
            ex["ex_pos_f_ids"] = ex_pos_f_ids

        for ex in examples:
            anchor_f_ids = []
            for f_id in ex["ex_pos_f_ids"]:
                # need at least one other positive example for one f_id for each example,
                # otherwise no anchor and positive possible
                if len(self.pos_examples[f_id]) > 1:
                    anchor_f_ids.append(f_id)
            ex["anchor_f_ids"] = anchor_f_ids
            if len(anchor_f_ids) > 0:
                self.examples.append(ex)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        anchor = self.examples[idx]
        # TODO allow for multiple positive and negative samples?
        pos, f_id = self.sample_positive(anchor)
        # TODO consider proportion of hard and soft negatives
        neg = self.sample_negative(f_id)

        pos_pair = self.create_pair_example(anchor, pos)
        neg_pair = self.create_pair_example(anchor, neg)

        example = {"ids": anchor["id"], "pos_examples": [pos_pair], "neg_examples": [neg_pair]}

        return example

    def create_pair_example(self, anchor, other):
        anchor_id = anchor["id"]
        other_id = other["id"]
        token_data = self.tokenizer(anchor["text"], other["text"])
        example = {
            "ids": f"{anchor_id}|{other_id}",
            "input_ids": token_data["input_ids"],
            "attention_mask": token_data["attention_mask"],
        }
        if "token_type_ids" in token_data:
            example["token_type_ids"] = token_data["token_type_ids"]
        return example

    def sample_positive(self, anchor):
        anchor_id = anchor["id"]
        anchor_f_ids = anchor["anchor_f_ids"]
        ex = None
        ex_id = anchor_id
        f_id = None

        while ex_id == anchor_id:
            # pick random framing
            f_idx = torch.randint(
                high=len(anchor_f_ids),
                size=[1],
            ).tolist()[0]
            f_id = anchor_f_ids[f_idx]
            f_examples = self.pos_examples[f_id]
            ex_idx = torch.randint(
                high=len(f_examples),
                size=[1],
            ).tolist()[0]
            ex = f_examples[ex_idx]
            ex_id = ex["id"]
        return ex, f_id

    def sample_negative(self, f_id):
        f_examples = self.neg_examples[f_id]
        ex_idx = torch.randint(
            high=len(f_examples),
            size=[1],
        ).tolist()[0]
        ex = f_examples[ex_idx]
        return ex

    def worker_init_fn(self, _):
        pass


class ContrastiveQuestionDataset(Dataset):
    examples: List[Dict[Any, Union[Any, Dict]]]

    def __init__(
        self,
        data_path: Union[str, List[str]],
        label_name: str,
        tokenizer,
        preprocess_config: TweetPreprocessConfig,
        predict_mode: str = "same",
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.label_name = label_name
        self.preprocess_config = preprocess_config
        self.predict_mode = predict_mode

        self.examples = []

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

    def read_path(self, data_path, stage=0):
        question_examples = defaultdict(list)
        all_examples = []
        for ex in read_jsonl(data_path):
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            ex_text = ex_text.strip().replace("\r", " ").replace("\n", " ")
            ex_text = preprocess_tweet(ex_text, self.preprocess_config)
            ex["text"] = ex_text
            ex_questions = ex[self.label_name]
            for q_id, q_info in ex_questions.items():
                question_examples[q_id].append(ex)
            all_examples.append(ex)
        if self.predict_mode == "same":
            for q_id, q_examples in question_examples.items():
                for anchor, other in itertools.combinations(q_examples, 2):
                    ex_pair = self.create_pair_example(anchor, other)
                    example = {"ids": ex_pair["ids"], "pos_examples": [ex_pair], "neg_examples": []}
                    self.examples.append(example)
        elif self.predict_mode == "other" or self.predict_mode == "all":
            for anchor, other in itertools.combinations(all_examples, 2):
                if self.predict_mode == "other":
                    anchor_qs = set(list(anchor[self.label_name].keys()))
                    other_qs = set(list(other[self.label_name].keys()))
                    # if both from the same question then skip
                    if len(anchor_qs.intersection(other_qs)) > 0:
                        continue
                ex_pair = self.create_pair_example(anchor, other)
                example = {"ids": ex_pair["ids"], "pos_examples": [ex_pair], "neg_examples": []}
                self.examples.append(example)
        else:
            raise ValueError(f"Unknown predict mode: {self.predict_mode}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]

        return example

    def create_pair_example(self, anchor, other):
        anchor_id = anchor["id"]
        other_id = other["id"]
        token_data = self.tokenizer(anchor["text"], other["text"])
        example = {
            "ids": f"{anchor_id}|{other_id}",
            "input_ids": token_data["input_ids"],
            "attention_mask": token_data["attention_mask"],
        }
        if "token_type_ids" in token_data:
            example["token_type_ids"] = token_data["token_type_ids"]
        return example

    def worker_init_fn(self, _):
        pass


@DATAMODULE_REGISTRY
class ContrastiveFrameDataModule(BaseDataModule):
    def __init__(
        self,
        label_name: str,
        frame_path: Union[str, List[str]] = None,
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        test_path: Union[str, List[str]] = None,
        predict_path: Union[str, List[str]] = None,
        predict_mode: str = "same",
        preprocess_config: TweetPreprocessConfig = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if preprocess_config is None:
            preprocess_config = TweetPreprocessConfig()
        self.preprocess_config = preprocess_config

        self.label_name = label_name
        self.predict_mode = predict_mode
        self.frame_path = frame_path
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path

        if self.train_path is not None:
            self.train_dataset = ContrastiveFrameDataset(
                tokenizer=self.tokenizer,
                data_path=self.train_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                preprocess_config=preprocess_config,
            )
        if self.val_path is not None:
            self.val_dataset = ContrastiveFrameDataset(
                tokenizer=self.tokenizer,
                data_path=self.val_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                preprocess_config=preprocess_config,
            )
        if self.test_path is not None:
            self.test_dataset = ContrastiveFrameDataset(
                tokenizer=self.tokenizer,
                data_path=self.test_path,
                frame_path=self.frame_path,
                label_name=self.label_name,
                preprocess_config=preprocess_config,
            )
        if self.predict_path is not None:
            self.predict_dataset = ContrastiveQuestionDataset(
                tokenizer=self.tokenizer,
                data_path=self.predict_path,
                label_name=self.label_name,
                preprocess_config=preprocess_config,
                predict_mode=self.predict_mode,
            )

    def create_collator(self):
        return ContrastiveFrameBatchCollator(
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )
