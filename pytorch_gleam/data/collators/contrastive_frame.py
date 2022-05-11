import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class ContrastiveFrameBatchCollator(BatchCollator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _calculate_multi_seq_padding(self, examples):
        if self.use_tpus:
            pad_seq_len = self.max_seq_len
        else:
            pad_seq_len = 0
            for ex in examples:
                ex_seqs = ex["pos_examples"] + ex["neg_examples"]
                for ex_seq in ex_seqs:
                    pad_seq_len = max(pad_seq_len, min(len(ex_seq["input_ids"]), self.max_seq_len))
        return pad_seq_len

    def __call__(self, examples):
        pad_seq_len = self._calculate_multi_seq_padding(examples)
        num_examples = len(examples)
        pos_samples = len(examples[0]["pos_examples"])
        neg_samples = len(examples[0]["neg_examples"])
        num_sequences_per_example = pos_samples + neg_samples
        # pos_samples + neg_samples
        num_sequences = num_examples * num_sequences_per_example

        input_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
        attention_mask = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
        token_type_ids = torch.zeros([num_examples, num_sequences_per_example, pad_seq_len], dtype=torch.long)
        ids = []
        p_ids = []
        n_ids = []
        has_token_type_ids = True
        for ex_idx, ex in enumerate(examples):
            ids.append(ex["ids"])
            ex_seqs = ex["pos_examples"] + ex["neg_examples"]
            for seq_idx, seq in enumerate(ex_seqs):
                self.pad_and_apply_seq(seq["input_ids"], input_ids, ex_idx, seq_idx)
                self.pad_and_apply_seq(seq["attention_mask"], attention_mask, ex_idx, seq_idx)
                if "token_type_ids" in seq:
                    self.pad_and_apply_seq(seq["token_type_ids"], token_type_ids, ex_idx, seq_idx)
                else:
                    has_token_type_ids = False
            for p_ex in ex["pos_examples"]:
                p_ids.append(p_ex["ids"])
            for n_ex in ex["neg_examples"]:
                n_ids.append(n_ex["ids"])
        batch = {
            "ids": ids,
            "p_ids": p_ids,
            "num_examples": num_examples,
            "pos_samples": pos_samples,
            "neg_samples": neg_samples,
            "pad_seq_len": pad_seq_len,
            "num_sequences_per_example": num_sequences_per_example,
            "num_sequences": num_sequences,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if has_token_type_ids:
            batch["token_type_ids"] = token_type_ids
        return batch

    def pad_and_apply_seq(self, id_list, id_tensor, ex_idx, seq_idx):
        ex_ids = id_list[: self.max_seq_len]
        id_tensor[ex_idx, seq_idx, : len(ex_ids)] = torch.tensor(ex_ids, dtype=torch.long)
