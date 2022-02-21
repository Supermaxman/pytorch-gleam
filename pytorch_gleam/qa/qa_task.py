import random
import re
from collections import defaultdict
from string import ascii_lowercase
from typing import Dict, List, Optional
from tqdm import tqdm

from datasets import load_dataset

import torch
from torch import nn
from transformers import AutoTokenizer


class QATaskConfig:
    def __init__(
        self,
        choices: Dict[str, int],
        label_map: Dict[str, int],
        path: str,
        prompt: str,
        split: Dict[str, str],
        template: str,
        max_size: int = -1,
        name: Optional[str] = None,
    ):
        self.choices = choices
        self.label_map = label_map
        self.name = name
        self.path = path
        self.prompt = prompt
        self.split = split
        self.template = template
        self.max_size = max_size


class MultiQATaskConfig:
    def __init__(self, tasks: List[QATaskConfig]):
        self.tasks = tasks


class QATaskModule(nn.Module):
    def __init__(self, tokenizer, config: QATaskConfig):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.template = self.config.template
        self.choice_map = self.config.choices
        self.choices = list(self.choice_map.keys())
        self.label_map = self.config.label_map
        self.inv_label_map = {v: k for k, v in self.label_map.items()}
        self.inv_choice_map = {v: k for k, v in self.choice_map.items()}

        self.choices_text = " ".join(
            [
                f"({o_letter}) {o_text}"
                for o_letter, o_text in zip(ascii_lowercase, self.choices)
            ]
        )
        self.pattern_keys = list(set(re.findall(r"\{\w*\}", self.template)))
        self.pattern = re.compile(
            "|".join(
                [re.escape(k) for k in sorted(self.pattern_keys, key=len, reverse=True)]
            ),
            flags=re.DOTALL,
        )
        self.data_keys = [
            (x[1:-1], x)
            for x in self.pattern_keys
            if x not in {"{prompt}", "{choices}"}
        ]
        ds_name = self.config.name
        ds_path = self.config.path
        if ds_name is not None:
            ds_path = f"{ds_path}|{ds_name}"
        self.path = ds_path

    def load(self, data_path: str, split: str):
        examples = []
        if split not in self.config.split:
            return examples
        ds = load_dataset(
            path=self.config.path,
            name=self.config.name,
            split=self.config.split[split],
            cache_dir=data_path,
        )
        for ds_idx, ex in tqdm(
            enumerate(ds), total=len(ds), desc=f"Loading {self.path} {split}"
        ):
            rep_dict = {
                "{prompt}": self.config.prompt,
                "{choices}": self.choices_text,
            }
            idx = ex["idx"] if "idx" in ex else ds_idx
            for sub_key, key in self.data_keys:
                if sub_key != "label" and sub_key != "idx":
                    value = ex[sub_key]
                    rep_dict[key] = value
            ex_id = f"{self.path}||{idx}"
            ex_text = self.pattern.sub(lambda x: rep_dict[x.group(0)], self.template)
            ex_text = ex_text.lower()
            ex_label = ex["label"]
            token_data = self.tokenizer(ex_text, truncation=True)
            input_ids = token_data["input_ids"]
            attention_mask = token_data["attention_mask"]
            choice_text = self.inv_choice_map[ex_label]
            choice_text = choice_text.lower()
            # TODO any more preprocessing, like urls
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

    def forward(self, qa_ids, qa_responses):
        # List[str]
        qa_response_texts = self.tokenizer.batch_decode(
            qa_responses, skip_special_tokens=True
        )
        preds = []
        for qa_response in qa_response_texts:
            qa_response = qa_response.title()
            if qa_response not in self.choice_map:
                qa_response = self.choices[-1]
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
        for ex_idx, qa_id in enumerate(qa_ids):
            ds_path, ex_id = qa_id.split("||")
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
