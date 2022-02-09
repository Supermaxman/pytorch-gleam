import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class SequenceToSequenceBatchCollator(BatchCollator):
    def __init__(self, max_label_seq_len: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_label_seq_len = max_label_seq_len

    def __call__(self, examples: list) -> dict:
        pad_seq_len = self._calculate_seq_padding(examples)
        label_pad_seq_len = self._calculate_seq_padding(
            examples, key="label_ids", max_seq_len=self.max_label_seq_len
        )

        batch_size = len(examples)
        input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
        attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)

        # [batch_size, labels_pad_seq_len]
        # -100 is ignored (treated as a mask) by loss and sequence generation
        has_labels = False
        labels = torch.zeros([batch_size], dtype=torch.long)
        label_ids = torch.ones([batch_size, label_pad_seq_len], dtype=torch.long) * -100
        ids = []
        for ex_idx, ex in enumerate(examples):
            ids.append(ex["ids"])
            self.pad_and_apply(ex["input_ids"], input_ids, ex_idx)
            self.pad_and_apply(ex["attention_mask"], attention_mask, ex_idx)
            if "label" in ex:
                has_labels = True
                labels[ex_idx] = ex["label"]
                self.pad_and_apply(ex["label_ids"], label_ids, ex_idx)

        batch = {
            "ids": ids,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if has_labels:
            batch["label_ids"] = label_ids
            batch["labels"] = labels
        return batch
