import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class ContrastiveChannelBatchCollator(BatchCollator):
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def _calculate_multi_seq_padding(self, examples):
        if self.use_tpus:
            pad_seq_len = self.max_seq_len
        else:
            pad_seq_len = 0
            for ex in examples:
                ex_seqs = [ex["t_ex"], ex["m_ex"]] + ex["p_samples"] + ex["n_samples"]
                for ex_seq in ex_seqs:
                    pad_seq_len = max(pad_seq_len, min(len(ex_seq["input_ids"]), self.max_seq_len))
        return pad_seq_len

    def __call__(self, examples):
        pos_samples = len(examples[0]["p_samples"])
        neg_samples = len(examples[0]["n_samples"])
        num_examples = len(examples)
        num_sequences_per_example = pos_samples + neg_samples
        # pos_samples + neg_samples
        num_sequences = num_examples * num_sequences_per_example
        # labels = torch.zeros([num_examples, num_sequences_per_example - 1], dtype=torch.long)
        # stages = torch.zeros([num_examples, num_sequences_per_example - 1], dtype=torch.long)

        # prompt which will be static, and not measured for model probs
        texts = []
        # target texts which will be measured for model probs
        text_targets = []
        relations = []
        directions = []
        labels = []
        stages = []
        ids = []
        m_ids = []
        s_ids = []
        assert pos_samples + neg_samples == len(examples[0]["relation_texts"])
        assert pos_samples + neg_samples == len(examples[0]["directions"])
        for ex in examples:
            ids.append(ex["t_ex"]["t_id"])
            m_ids.append(ex["m_ex"]["m_id"])
            # "t_ex": t_ex,
            # "m_ex": m_ex,
            # "labels": labels,
            # "stages": stages,
            # "p_samples": pos_samples,
            # "n_samples": neg_samples,
            # "direction": direction,
            # "relations": relations,
            # "relation_texts": relation_texts,
            ex_text = ex["t_ex"]["t_text"]
            m_text = ex["m_ex"]["m_text"]
            samples = ex["p_samples"] + ex["n_samples"]
            relations.extend(ex["relations"])
            directions.extend(ex["directions"])
            labels.extend(ex["labels"])
            stages.extend(ex["stages"])

            for s_sample, s_rel_text, s_direction in zip(samples, ex["relation_texts"], ex["directions"]):
                s_text = s_sample["t_text"]
                texts.append(f"Misinformation:\n{m_text}\nRelation:\n{s_rel_text}\n")
                s_ids.append(s_sample["t_id"])
                if s_direction == 0:
                    text_targets.append(f"First Text:\n{ex_text}\nSecond Text:\n{s_text}\n")
                else:
                    text_targets.append(f"First Text:\n{s_text}\nSecond Text:\n{ex_text}\n")

        model_inputs = self.tokenizer(
            text=texts,
            padding="max_length" if self.use_tpus else "longest",
            truncation="longest_first",
            max_length=self.max_seq_len,
            return_tensors="pt",
        )

        model_targets = self.tokenizer(
            text=text_targets,
            padding="max_length" if self.use_tpus else "longest",
            truncation="longest_first",
            max_length=self.max_seq_len,
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
            "pos_samples": pos_samples,
            "neg_samples": neg_samples,
            "num_sequences_per_example": num_sequences_per_example,
            "num_sequences": num_sequences,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "target_ids": target_ids,
            "relations": torch.tensor(relations, dtype=torch.long),
            "directions": torch.tensor(directions, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "stages": torch.tensor(stages, dtype=torch.long),
        }
        return batch
