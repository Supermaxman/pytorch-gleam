import torch

from pytorch_gleam.data.collators.base_collators import BatchCollator


class DirectACSBatchCollator(BatchCollator):
    def __init__(self, tokenizer, *args, **kwargs):
        self.tokenizer = tokenizer
        super().__init__(*args, **kwargs)

    def __call__(self, examples):
        pos_samples = len(examples[0]["p_samples"])
        num_examples = len(examples)
        num_sequences_per_example = pos_samples
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
        for ex in examples:
            ids.append(ex["t_ex"]["t_id"])
            m_ids.append(ex["m_ex"]["m_id"])
            ex_text = ex["t_ex"]["t_text"]
            m_text = ex["m_ex"]["m_text"]
            samples = ex["p_samples"]
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
                texts.append(f"mnli hypothesis: {m_text} premise: {given_text} target: {target_text}")
                text_targets.append(s_rel_text)

        model_inputs = self.tokenizer(
            text=texts,
            padding="max_length" if self.use_tpus else "longest",
            # can't afford to truncate here
            truncation="longest_first",
            # m_text and given text get 2/3 seq len
            max_length=self.max_seq_len - 10,
            return_tensors="pt",
        )
        model_targets = self.tokenizer(
            text=text_targets,
            padding="max_length" if self.use_tpus else "longest",
            truncation="longest_first",
            # target text gets 1/3 seq len
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
            "pos_samples": pos_samples,
            "num_sequences_per_example": num_sequences_per_example,
            "num_sequences": num_sequences,
            "relations": torch.tensor(relations, dtype=torch.long),
            "directions": torch.tensor(directions, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "stages": torch.tensor(stages, dtype=torch.long),
        }
        return batch
