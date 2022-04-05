import dataclasses
from typing import List, Union

from pytorch_gleam.data.collators import BertPreBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule, BaseIterableDataset


@dataclasses.dataclass
class BertPreTrainDataConfig:
    masked_lm_prob: float = 0.15
    short_seq_prob: float = 0.10
    max_seq_length: int = 128
    max_predictions_per_seq: int = 20
    dupe_factor: int = 10
    do_whole_word_mask: bool = True


class BertPreDataModule(BaseDataModule):
    def __init__(
        self,
        worker_estimate: int,
        train_examples: int,
        val_examples: int,
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.worker_estimate = worker_estimate
        self.train_examples = train_examples
        self.val_examples = val_examples

        self.train_path = train_path
        self.val_path = val_path

        if self.train_path is not None:
            self.train_dataset = BaseIterableDataset(
                num_examples=self.train_examples, worker_estimate=self.worker_estimate
            )
            self.train_dataset.load(self.train_path)
        if self.val_path is not None:
            self.val_dataset = BaseIterableDataset(num_examples=self.val_examples, worker_estimate=self.worker_estimate)
            self.val_dataset.load(self.val_path)

    def create_collator(self):
        return BertPreBatchCollator(
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )
