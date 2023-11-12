import torch


class MultiClassFrameValuesBatchCollator:
    def __init__(self, tokenizer, values_map, max_seq_len: int = 512, use_tpus=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.values_map = values_map
        self.max_seq_len = max_seq_len
        self.use_tpus = use_tpus
        self.num_values = len(values_map)

    def __call__(self, examples: list) -> dict:
        batch_size = len(examples)
        # [ex_count, num_classes]
        labels = torch.zeros([batch_size], dtype=torch.long)
        value_mask = torch.zeros([batch_size, self.num_values], dtype=torch.long)
        ids = []
        texts = []
        for ex_idx, ex in enumerate(examples):
            ids.append(ex["ids"])
            if "label" in ex:
                labels[ex_idx] = ex["label"]
            texts.append(ex["text"])
            value_mask[ex_idx, ex["value_ids"]] = 1
        data = self.tokenizer(
            text=texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        )
        batch = {
            "ids": ids,
            "input_ids": data["input_ids"],
            "attention_mask": data["attention_mask"],
            "value_mask": value_mask,
            "labels": labels,
        }
        if "token_type_ids" in data:
            batch["token_type_ids"] = data["token_type_ids"]
        return batch
