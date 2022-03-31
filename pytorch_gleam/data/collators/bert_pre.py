from time import sleep

import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class BertPreBatchCollator(BatchCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, examples: list) -> dict:
        print("START")
        sleep(10)
        print(len(examples))
        sleep(10)
        print(examples[0])
        sleep(10)
        print(examples)
        sleep(10)
        print(len(examples[0]))
        sleep(10)
        print("END")
        sleep(10)
        # if we do pre-batching in the dataset object then this is the pattern
        if isinstance(examples[0], list) and len(examples) == 1:
            examples = examples[0]

        pad_seq_len = self._calculate_seq_padding(examples)

        batch_size = len(examples)
        # "input_ids": input_ids,
        # "attention_mask": [1] * len(input_ids),
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
            attention_mask[ex_idx, : len(ex["input_ids"])] = 1
            type_0_len, type_1_len = ex["segment_lengths"]
            token_type_ids[ex_idx, type_0_len : type_0_len + type_1_len] = 1
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
