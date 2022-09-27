import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class DirectStanceBatchCollator(BatchCollator):
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def __call__(self, examples):
        num_examples = len(examples)

        # prompt which will be static, and not measured for model probs
        texts = []
        # target texts which will be measured for model probs
        text_targets = []
        labels = []
        stages = []
        ids = []
        m_ids = []
        s_ids = []
        m_label_idxs = []
        for ex in examples:
            ids.append(ex["ex_id"])
            m_ids.append(ex["m_id"])
            labels.append(ex["m_label"])
            stages.append(ex["stage"])
            m_label_idxs.append(ex["m_label_idx"])
            ex_text = ex["ex_text"]
            m_text = ex["m_text"]
            r_text = ex["m_label_text"]
            texts.append(f"mnli hypothesis: {m_text} premise: {ex_text}")
            text_targets.append(r_text)

        model_inputs = self.tokenizer(
            text=texts,
            padding="max_length" if self.use_tpus else "longest",
            # can't afford to truncate here
            truncation="longest_first",
            max_length=self.max_seq_len - 10,
            return_tensors="pt",
        )
        model_targets = self.tokenizer(
            text=text_targets,
            padding="max_length" if self.use_tpus else "longest",
            truncation="longest_first",
            max_length=10,
            return_tensors="pt",
        )
        target_ids = model_targets["input_ids"]
        # replace padding token id's of the labels by -100 so it's ignored by the loss
        # https://huggingface.co/docs/transformers/v4.22.1/en/model_doc/t5#overview
        target_ids[target_ids == self.tokenizer.pad_token_id] = -100

        batch = {
            "ids": ids,
            "m_ids": m_ids,
            "s_ids": s_ids,
            "num_examples": num_examples,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "target_ids": target_ids,
            "label_idxs": torch.tensor(m_label_idxs, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "stages": torch.tensor(stages, dtype=torch.long),
        }
        return batch
