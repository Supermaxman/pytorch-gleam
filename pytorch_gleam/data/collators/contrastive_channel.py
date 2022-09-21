import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class ContrastiveChannelBatchCollator(BatchCollator):
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def __call__(self, examples):
        pos_samples = len(examples[0]["p_samples"])
        neg_samples = len(examples[0]["n_samples"])
        num_examples = len(examples)
        num_sequences_per_example = pos_samples + neg_samples
        # pos_samples + neg_samples
        num_sequences = num_examples * num_sequences_per_example

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
                s_ids.append(s_sample["t_id"])
                given_text = ex_text
                target_text = s_text
                if s_direction == 1:
                    given_text = s_text
                    target_text = ex_text
                texts.append(f"Topic:\n{m_text}\nRelation:\n{s_rel_text}\nText:\n{given_text}\n")
                text_targets.append(f"Relation Text:\n{target_text}\n")

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
        for input_ids, target_ids in zip(model_inputs["input_ids"], model_targets["input_ids"]):
            x = input_ids[input_ids != self.tokenizer.pad_token_id]
            y = target_ids[target_ids != self.tokenizer.pad_token_id]
            lines = [
                f"Input Ids: {len(x)} Target Ids: {len(y)}",
                self.tokenizer.decode(x),
                "---------------------------",
                self.tokenizer.decode(y),
                "===========================",
            ]
            print("\n".join(lines))
            input()

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
