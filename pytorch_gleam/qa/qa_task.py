import random
import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, List, Optional, Union

import torch
import ujson as json
from datasets import load_dataset
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer

from pytorch_gleam.modeling.metrics import Metric


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


class QATaskPrompt:
    def __init__(self, text: Union[str, Dict[str, Optional[str]]], choices: List[Dict[str, int]]):
        if isinstance(text, dict):
            text = list(text.keys())[0]
        self.text = text
        self.choices = choices
        self.inv_choice_lists = {}
        self.choice_map = {}
        for choices in self.choices:
            for c_name, c_idx in choices.items():
                self.choice_map[c_name] = c_idx
                if c_idx not in self.inv_choice_lists:
                    self.inv_choice_lists[c_idx] = []
                self.inv_choice_lists[c_idx].append(c_name)
        self.default_choice = self.inv_choice_lists[max(self.inv_choice_lists.keys())][0]

    def __str__(self):
        choices_txt = []
        for choices in self.choices:
            choices_txt.append("|".join([f"{k}-({str(v)})" for k, v in choices.items()]))
        choices_txt = "|".join([f"({x})" for x in choices_txt])
        config_text = "|".join([f"choices-({choices_txt})", f"text-({self.text})"])
        return config_text


class QATaskConfig:
    def __init__(
        self,
        label_map: Dict[str, int],
        path: str,
        prompts: List[QATaskPrompt],
        split: Dict[str, str],
        template: str,
        metric: Metric,
        task: str,
        location: Optional[str] = None,
        label_name: Optional[str] = None,
        max_size: int = -1,
        name: Optional[str] = None,
        num_samples: int = 1,
    ):
        self.label_map = label_map
        self.name = name
        self.path = path
        self.prompts = prompts
        self.split = split
        self.template = template
        self.max_size = max_size
        self.metric = metric
        self.task = task
        self.location = location
        self.label_name = label_name
        self.num_samples = num_samples

    def __str__(self):
        prompts_txt = "|".join([f"({str(p)})" for p in self.prompts])
        label_map_txt = "|".join([f"{k}-({str(v)})" for k, v in self.label_map.items()])
        split_txt = "|".join([f"{k}-({str(v)})" for k, v in self.split.items()])
        config_text = "|".join(
            [
                f"label_map-({label_map_txt})",
                f"path-({self.path})",
                f"prompts-({prompts_txt})",
                f"split-({split_txt})",
                f"template-({self.template})",
                f"max_size-({self.max_size})",
                f"name-({self.name})",
            ]
        )
        return config_text


class MultiQATaskConfig:
    def __init__(self, tasks: List[QATaskConfig]):
        self.tasks = tasks

    def __str__(self):
        return "|".join(f"({t})" for t in self.tasks)


class QATaskModule(nn.Module):
    def __init__(self, tokenizer, config: QATaskConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.template = self.config.template
        self.prompts = self.config.prompts
        self.metric = self.config.metric
        self.label_map = self.config.label_map
        self.label_name = self.config.label_name
        self.location = self.config.location
        self.task = self.config.task
        self.split = self.config.split
        self.num_samples = self.num_samples
        self.inv_label_map = {v: k for k, v in self.label_map.items()}

        self.pattern_keys = list(set(re.findall(r"\{\w*\}", self.template)))
        self.pattern = re.compile(
            "|".join([re.escape(k) for k in sorted(self.pattern_keys, key=len, reverse=True)]),
            flags=re.DOTALL,
        )
        self.data_keys = [(x[1:-1], x) for x in self.pattern_keys if x not in {"{prompt}", "{choices}"}]
        ds_name = self.config.name
        ds_path = self.config.path
        if ds_name is not None:
            ds_path = f"{ds_path}|{ds_name}"
        self.path = ds_path

        self.choice_map = {}
        for prompt in self.prompts:
            for c_name, c_idx in prompt.choice_map.items():
                self.choice_map[c_name] = c_idx

        self.default_choice = self.prompts[0].default_choice

    def load(self, data_path: str, split: str):
        examples = []
        if split not in self.config.split:
            return examples
        if self.location is not None and self.location == "local":
            ds = self.load_local_dataset(split=split)
        else:
            ds = load_dataset(
                path=self.config.path,
                name=self.config.name,
                split=self.config.split[split],
                cache_dir=data_path,
            )
        for s_idx in range(self.config.num_samples):
            for ds_idx, ex in tqdm(enumerate(ds), total=len(ds), desc=f"Loading {self.path} {split}"):
                ex_label = ex["label"]
                if ex_label < 0:
                    continue

                # random prompt with aligned choices
                prompt = random.choice(self.prompts)

                inv_choices_map = {}
                # pick a random choice for each label idx
                for inv_idx, choice_list in prompt.inv_choice_lists.items():
                    inv_choices_map[inv_idx] = random.choice(choice_list)

                choices = list(inv_choices_map.values())
                # random order of choices
                random.shuffle(choices)

                choices_text = " ".join(
                    [f"({o_letter}) {o_text}" for o_letter, o_text in zip(ascii_lowercase, choices)]
                )
                rep_dict = {
                    "{prompt}": prompt.text,
                    "{choices}": choices_text,
                }
                idx = ex["idx"] if "idx" in ex else ds_idx
                if self.num_samples > 1:
                    idx = f"{idx}-s{s_idx}"
                for sub_key, key in self.data_keys:
                    if sub_key != "label" and sub_key != "idx":
                        value = ex[sub_key]
                        rep_dict[key] = value
                ex_id = f"{self.path}||{idx}"
                ex_text = self.pattern.sub(lambda x: rep_dict[x.group(0)], self.template)
                ex_text = ex_text.lower()
                # TODO any more preprocessing, like urls
                token_data = self.tokenizer(ex_text, truncation=True)
                input_ids = token_data["input_ids"]
                attention_mask = token_data["attention_mask"]
                choice_text = inv_choices_map[ex_label]
                choice_text = choice_text.lower()
                label_data = self.tokenizer(choice_text)
                label_ids = label_data["input_ids"]
                example = {
                    "ids": ex_id,
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "label_ids": label_ids,
                    "label": ex_label,
                }
                examples.append(example)

        if self.config.max_size > 0:
            random.shuffle(examples)
            examples = examples[: self.config.max_size]
        return examples

    def load_local_dataset(self, split):
        split_file = self.split[split]
        frames = None
        if "frames" in self.split:
            with open(self.split["frames"]) as f:
                frames = json.load(f)
        ds = []
        skipped_labels = set()
        for ex in read_jsonl(split_file):
            ex_id = str(ex["id"])
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]

            labels = ex[self.label_name]
            if isinstance(labels, dict):
                for f_id, f_label in labels.items():
                    frame = frames[f_id]
                    frame_text = frame["text"]
                    if f_label not in self.label_map:
                        if f_label not in skipped_labels:
                            print(f"Skipping Label {f_label} in {self.path}")
                            skipped_labels.add(f_label)
                        continue
                    row = {
                        "idx": f"{ex_id}|{f_id}",
                        "text": ex_text,
                        "frame": frame_text,
                        "label": self.label_map[f_label],
                    }
                    ds.append(row)
            else:
                for frame in labels:
                    f_id = frame["misconception_id"]
                    frame_text = frame["misconception_text"]
                    f_label = frame["label"]
                    if f_label not in self.label_map:
                        continue
                    row = {
                        "idx": f"{ex_id}|{f_id}",
                        "text": ex_text,
                        "frame": frame_text,
                        "label": self.label_map[f_label],
                    }
                    ds.append(row)
        return ds

    def forward(self, qa_ids, qa_responses):
        # List[str]
        qa_response_texts = self.tokenizer.batch_decode(qa_responses, skip_special_tokens=True)
        preds = []
        for qa_response in qa_response_texts:
            qa_response = qa_response.title()
            if qa_response not in self.choice_map:
                qa_response = self.default_choice
            qa_pred = self.choice_map[qa_response]
            preds.append(qa_pred)
        preds = torch.tensor(preds, dtype=torch.long)
        return qa_ids, preds


class MultiQATaskModule(nn.Module):
    def __init__(self, tokenizer_name: str, config: MultiQATaskConfig):
        super().__init__()
        self.tokenizer_name = tokenizer_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.config = config
        self.datasets = {}
        for ds_config in self.config.tasks:
            ds = QATaskModule(self.tokenizer, ds_config)
            self.datasets[ds.path] = ds

    def __str__(self):
        return f"{type(self)}tokenizer_name-({self.tokenizer_name})|config-({self.config})"

    def load(self, data_path: str, split: str):
        examples = []
        for ds_name, ds in tqdm(
            self.datasets.items(),
            total=len(self.datasets),
            desc=f"Loading {split} datasets",
        ):
            examples.extend(ds.load(data_path, split))
        return examples

    def forward(self, qa_ids, qa_responses):
        # ds_path||ex_id
        ds_ids = defaultdict(list)
        ds_indices = defaultdict(list)
        for ex_idx, ex_id in enumerate(qa_ids):
            ds_path, ds_idx = ex_id.split("||")
            ds_ids[ds_path].append(ex_id)
            ds_indices[ds_path].append(ex_idx)

        f_ids = []
        f_preds = []
        for ds_path, ds in self.datasets.items():
            d_ids = ds_ids[ds_path]
            d_indices = torch.tensor(ds_indices[ds_path], dtype=torch.long)
            d_qa_responses = qa_responses[d_indices]
            d_ids, d_preds = ds(d_ids, d_qa_responses)
            f_ids.extend(d_ids)
            f_preds.append(d_preds)
        f_preds = torch.cat(f_preds, dim=0)
        return f_ids, f_preds

    def calculate_metrics(self, qa_ids, qa_labels, qa_preds):
        # ds_path||ex_id
        ds_ids = defaultdict(list)
        ds_indices = defaultdict(list)
        for ex_idx, ex_id in enumerate(qa_ids):
            ds_path, ds_idx = ex_id.split("||")
            ds_ids[ds_path].append(ex_id)
            ds_indices[ds_path].append(ex_idx)

        ds_metrics = {}
        for ds_path, ds in self.datasets.items():
            # d_ids = ds_ids[ds_path]
            d_indices = torch.tensor(ds_indices[ds_path], dtype=torch.long)
            d_qa_labels = qa_labels[d_indices]
            d_qa_preds = qa_preds[d_indices]
            d_metrics = ds.metric(d_qa_labels, d_qa_preds)
            ds_metrics[ds_path] = d_metrics
        return ds_metrics
