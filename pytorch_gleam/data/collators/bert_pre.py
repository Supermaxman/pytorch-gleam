import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class BertPreBatchCollator(BatchCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, examples: list) -> dict:
        pad_seq_len = self._calculate_seq_padding(examples)

        batch_size = len(examples)
        # "input_ids": input_ids,
        # "attention_mask": [1] * len(input_ids),
        # "token_type_ids": list(instance["segment_ids"]),
        input_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
        attention_mask = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
        token_type_ids = torch.zeros([batch_size, pad_seq_len], dtype=torch.long)
        # "masked_lm_positions": masked_lm_positions,
        # "masked_lm_ids": masked_lm_ids,
        # -100 is ignored by loss function
        masked_lm_labels = -100 * torch.ones([batch_size, pad_seq_len], dtype=torch.long)

        next_sentence_labels = torch.zeros([batch_size], dtype=torch.long)
        for ex_idx, ex in enumerate(examples):
            self.pad_and_apply(ex["input_ids"], input_ids, ex_idx)
            self.pad_and_apply(ex["attention_mask"], attention_mask, ex_idx)
            self.pad_and_apply(ex["token_type_ids"], token_type_ids, ex_idx)
            for s_idx, s_id in zip(ex["masked_lm_positions"], ex["masked_lm_ids"]):
                masked_lm_labels[ex_idx, s_idx] = s_id

            next_sentence_labels[ex_idx] = ex["next_sentence_label"]

        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "masked_lm_labels": masked_lm_labels,
            "next_sentence_labels": next_sentence_labels,
        }

        return batch
