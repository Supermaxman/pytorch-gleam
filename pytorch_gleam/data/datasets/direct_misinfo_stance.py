import torch
from transformers import AutoTokenizer

from pytorch_gleam.data.collators import DirectStanceBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.datasets.misinfo_stance import MisinfoStanceDataset
from pytorch_gleam.data.twitter import TweetPreprocessConfig


class DirectMisinfoStanceDataset(MisinfoStanceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_text_map = {0: "No Stance", 1: "Accept", 2: "Reject"}
        for ex in self.examples:
            ex["m_label_idx"] = ex["m_label"]
            ex["m_label_text"] = self.label_text_map[ex["m_label"]]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ex = self.examples[idx]
        return ex


class DirectInferMisinfoStanceDataset(DirectMisinfoStanceDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_examples = self.examples
        self.examples = []
        for ex in self.base_examples:
            for i in range(3):
                new_ex = ex.copy()
                new_ex["m_label_idx"] = i
                new_ex["m_label_text"] = self.label_text_map[i]
                self.examples.append(new_ex)


class DirectMisinfoStanceDataModule(BaseDataModule):
    def __init__(
        self,
        misinfo_path: str,
        tokenizer_name: str,
        preprocess_config: TweetPreprocessConfig = None,
        label_name: str = "misinfo",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if preprocess_config is None:
            preprocess_config = TweetPreprocessConfig(do_lower_case=False)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.misinfo_path = misinfo_path
        self.label_name = label_name

        if self.train_path is not None:
            self.train_dataset = DirectMisinfoStanceDataset(
                data_path=self.train_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
                label_name=self.label_name,
            )
        if self.val_path is not None:
            self.val_dataset = DirectInferMisinfoStanceDataset(
                data_path=self.val_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
                label_name=self.label_name,
            )
        if self.test_path is not None:
            self.test_dataset = DirectInferMisinfoStanceDataset(
                data_path=self.test_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
                label_name=self.label_name,
            )
        if self.predict_path is not None:
            self.predict_dataset = DirectInferMisinfoStanceDataset(
                data_path=self.predict_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
                label_name=self.label_name,
            )

    def create_collator(self):
        return DirectStanceBatchCollator(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )
