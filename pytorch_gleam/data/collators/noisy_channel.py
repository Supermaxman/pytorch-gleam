import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class NoisyChannelBatchCollator(BatchCollator):
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
            labels.append(ex["labels"])
            stages.append(ex["stages"])
            m_label_idxs.append(ex["m_label_idx"])
            ex_text = ex["ex_text"]
            m_text = ex["m_text"]
            r_text = ex["m_label_text"]
            texts.append(f"mnli hypothesis: {m_text} relation: {r_text}")
            text_targets.append(ex_text)

        model_inputs = self.tokenizer(
            text=texts,
            padding="max_length" if self.use_tpus else "longest",
            # can't afford to truncate here
            truncation="longest_first",
            # m_text get 1/2 seq len
            max_length=self.max_seq_len // 2,
            return_tensors="pt",
        )
        model_targets = self.tokenizer(
            text=text_targets,
            padding="max_length" if self.use_tpus else "longest",
            truncation="longest_first",
            # target text gets 1/2 seq len
            max_length=self.max_seq_len // 2,
            return_tensors="pt",
        )
        # debugging
        # for input_ids, target_ids in zip(model_inputs["input_ids"], model_targets["input_ids"]):
        #     x = input_ids[input_ids != self.tokenizer.pad_token_id]
        #     y = target_ids[target_ids != self.tokenizer.pad_token_id]
        #     lines = [
        #         f"Input Ids: {len(x)} Target Ids: {len(y)}",
        #         self.tokenizer.decode(x),
        #         "---------------------------",
        #         self.tokenizer.decode(y),
        #         "===========================",
        #     ]
        #     print("\n".join(lines))
        #     input()

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
