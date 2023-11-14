import torch


class MultiClassFrameMultiValuesBatchCollator:
    def __init__(self, tokenizer, cultural_list, moral_list, max_seq_len: int = 512, use_tpus=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.cultural_list = cultural_list
        self.moral_list = moral_list
        self.max_seq_len = max_seq_len
        self.use_tpus = use_tpus
        self.num_cultural_values = len(cultural_list)
        self.num_moral_values = len(moral_list)

    def __call__(self, examples: list) -> dict:
        batch_size = len(examples)
        # [ex_count, num_classes]
        labels = torch.zeros([batch_size], dtype=torch.long)
        cultural_mask = torch.zeros([batch_size, self.num_cultural_values], dtype=torch.long)
        moral_mask = torch.zeros([batch_size, self.num_moral_values], dtype=torch.long)
        ids = []
        texts = []
        for ex_idx, ex in enumerate(examples):
            ids.append(ex["ids"])
            if "label" in ex:
                labels[ex_idx] = ex["label"]
            texts.append(ex["text"])
            cultural_mask[ex_idx, ex["cultural_value_ids"]] = 1
            moral_mask[ex_idx, ex["moral_value_ids"]] = 1
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
            "cultural_mask": cultural_mask,
            "moral_mask": moral_mask,
            "labels": labels,
        }
        if "token_type_ids" in data:
            batch["token_type_ids"] = data["token_type_ids"]
        return batch
