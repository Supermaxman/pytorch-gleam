import itertools
from collections import defaultdict

import torch
from transformers import AutoTokenizer

from pytorch_gleam.data.collators import ContrastiveChannelBatchCollator
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.datasets.misinfo_stance import MisinfoStanceDataset
from pytorch_gleam.data.twitter import TweetPreprocessConfig


class ContrastiveChannelMisinfoStanceDataset(MisinfoStanceDataset):
    def __init__(self, pos_samples: int = 1, neg_samples: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.relation_map = {0: "entailment", 1: "contradiction"}
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples
        self.permutations = [
            self.flip_polarity,
            self.flip_rel,
            self.zero_polarity,
            # self.zero_all_polarity
        ]
        self.full_examples = []
        self.base_examples = self.examples
        self.examples = []
        self.label_examples = defaultdict(lambda: defaultdict(list))
        self.m_examples = defaultdict(list)
        self.m_exs = {}
        for pair_ex in self.base_examples:
            ex_id = pair_ex["ex_id"]
            m_id = pair_ex["m_id"]
            m_label = pair_ex["m_label"]
            if m_id in self.m_exs:
                m_ex = self.m_exs[m_id]
            else:
                m_ex = {
                    "m_id": m_id,
                    "m_text": pair_ex["m_text"],
                }
                self.m_exs[m_id] = m_ex
            t_ex = {
                "t_id": ex_id,
                "t_text": pair_ex["ex_text"],
                "m_label": m_label,
                "stage": pair_ex["stage"],
            }
            ex = {
                "t_ex": t_ex,
                "m_ex": m_ex,
            }
            self.label_examples[m_id][m_label].append(t_ex)
            self.m_examples[m_id].append(t_ex)
            # no stance has no true pairs
            if m_label != 0:
                self.examples.append(ex)
            self.full_examples.append(ex)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ex = self.examples[idx]
        t_ex = ex["t_ex"]
        m_ex = ex["m_ex"]
        # 1 is accept
        # 2 is reject
        # 0 is no stance, and is only in negative_examples
        tm_stance = t_ex["m_label"]
        m_id = m_ex["m_id"]

        # positive examples
        # 0 is entail
        # 1 is contradict
        # TODO could sample positive relations too
        tmp_relation = self._sample_relation()
        pm_stance = tmp_stance(tmp_relation, tm_stance)
        pos_examples = self.label_examples[m_id][pm_stance]
        # if pos_examples is empty then flip tmp_relation
        if len(pos_examples) == 0:
            tmp_relation = flip_relation(tmp_relation)
            pm_stance = tmp_stance(tmp_relation, tm_stance)
            pos_examples = self.label_examples[m_id][pm_stance]

        # t - tmp_relation -> p
        pos_samples = self._sample(pos_examples, self.pos_samples, replacement=False)
        # negative sampling
        # Four ways to create negative samples
        # (flip polarity): a -> a to a -> r
        # (flip rel) a -> a to a \-> a
        # (zero polarity): a -> a to a -> ns
        # (zero all polarity) a -> a to ns -> ns
        # all four permutations could be useful for training
        # first three are necessary
        # fourth is only needed if ns will be used in seed kb for inference
        neg_relation_samples = self._negative_sample(m_id, tmp_relation, pm_stance, pos_samples, self.neg_samples)
        neg_samples = []
        neg_relations = []
        for neg_relation, neg_sample in neg_relation_samples:
            neg_samples.append(neg_sample)
            neg_relations.append(neg_relation)

        direction = self._sample_direction()
        # [pos sample relation labels + neg_sample_relation_labels]
        # not used for training, stance labels are unnecessary
        labels = [tm_stance] + [p_ex["m_label"] for p_ex in pos_samples] + [n_ex["m_label"] for n_ex in neg_samples]
        stages = [t_ex["stage"]] + [p_ex["stage"] for p_ex in pos_samples] + [n_ex["stage"] for n_ex in neg_samples]
        relations = [tmp_relation for _ in range(len(pos_samples))] + neg_relations
        directions = [direction for _ in range(len(pos_samples) + len(neg_samples))]
        relation_texts = [self.relation_map[rel] for rel in relations]
        ex = {
            "t_ex": t_ex,
            "m_ex": m_ex,
            "labels": labels,
            "stages": stages,
            "p_samples": pos_samples,
            "n_samples": neg_samples,
            "directions": directions,
            "relations": relations,
            "relation_texts": relation_texts,
        }

        return ex

    def _negative_sample(self, m_id, tmp_relation, pm_stance, pos_samples, sample_count):
        possible_permutations = [
            permutation
            for permutation in self.permutations
            if permutation(m_id, tmp_relation, pm_stance, pos_samples) is not None
        ]
        p_indices = torch.randint(
            high=len(possible_permutations),
            size=[sample_count],
        ).tolist()
        samples = [possible_permutations[i](m_id, tmp_relation, pm_stance, pos_samples) for i in p_indices]

        return samples

    def flip_polarity(self, m_id, tmp_relation, pm_stance, pos_samples):
        # (flip polarity): a -> a to a -> r
        flip_pm_stance = flip_tm_stance(pm_stance)
        m_examples = self.label_examples[m_id][flip_pm_stance]
        if len(m_examples) == 0:
            return None
        s_example = self._sample(m_examples, m_count=1, replacement=True)[0]
        return tmp_relation, s_example

    def flip_rel(self, m_id, tmp_relation, pm_stance, pos_samples):
        # (flip rel) a -> a to a \-> a
        flip_tmp_relation = flip_relation(tmp_relation)
        s_example = self._sample(pos_samples, m_count=1, replacement=True)[0]
        return flip_tmp_relation, s_example

    def zero_polarity(self, m_id, tmp_relation, pm_stance, pos_samples):
        # (zero polarity): a -> a to a -> ns
        # 0 is zero stance polarity
        m_examples = self.label_examples[m_id][0]
        if len(m_examples) == 0:
            return None
        s_example = self._sample(m_examples, m_count=1, replacement=True)[0]
        return tmp_relation, s_example

    def _sample(self, m_examples, m_count, replacement=False):
        samples = []
        if m_count <= 0:
            return samples
        if not replacement:
            m_s_indices = torch.randperm(
                n=len(m_examples),
            ).tolist()[:m_count]
        else:
            m_s_indices = torch.randint(
                high=len(m_examples),
                size=[m_count],
            ).tolist()
        for s_idx in m_s_indices:
            samples.append(m_examples[s_idx])
        return samples

    def _sample_direction(self):
        r = torch.rand(
            size=(1,),
        ).tolist()[0]
        if r < 0.5:
            return 0
        else:
            return 1

    def _sample_relation(self):
        r = torch.rand(
            size=(1,),
        ).tolist()[0]
        if r < 0.5:
            return 0
        else:
            return 1

    def worker_init_fn(self, _):
        pass


class ContrastiveChannelMisinfoInferStanceDataset(ContrastiveChannelMisinfoStanceDataset):
    def __init__(self, pair_count=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # TODO support pair counts greater than 1
        self.pair_count = pair_count
        self.examples = []
        for m_id, m_examples in self.m_examples.items():
            m_ex = self.m_exs[m_id]
            for a_ex, b_ex in itertools.combinations(m_examples, 2):
                pair_ex = {
                    "t_ex": a_ex,
                    "p_ex": b_ex,
                    "m_ex": m_ex,
                }
                self.examples.append(pair_ex)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        ex = self.examples[idx]
        t_ex = ex["t_ex"]
        m_ex = ex["m_ex"]
        # same pos example, but two possible relations
        pair_examples = [ex["p_ex"], ex["p_ex"]]
        # 1 is accept
        # 2 is reject
        # 0 is no stance, and is only in negative_examples

        # only one direction
        directions = [0 for _ in range(len(pair_examples))]
        labels = [t_ex["m_label"]] + [p_ex["m_label"] for p_ex in pair_examples]
        stages = [t_ex["stage"]] + [p_ex["stage"] for p_ex in pair_examples]
        # both relations (agree, disagree)
        relations = [0, 1]
        relation_texts = [self.relation_map[rel] for rel in relations]
        ex = {
            "t_ex": t_ex,
            "m_ex": m_ex,
            "labels": labels,
            "stages": stages,
            "p_samples": pair_examples,
            "n_samples": [],
            "directions": directions,
            "relations": relations,
            "relation_texts": relation_texts,
        }

        return ex


class ContrastiveChannelMisinfoStanceDataModule(BaseDataModule):
    def __init__(
        self,
        misinfo_path: str,
        tokenizer_name: str,
        pos_samples: int = 1,
        neg_samples: int = 1,
        preprocess_config: TweetPreprocessConfig = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if preprocess_config is None:
            preprocess_config = TweetPreprocessConfig(do_lower_case=False)

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.misinfo_path = misinfo_path
        self.pos_samples = pos_samples
        self.neg_samples = neg_samples

        if self.train_path is not None:
            self.train_dataset = ContrastiveChannelMisinfoStanceDataset(
                pos_samples=self.pos_samples,
                neg_samples=self.neg_samples,
                data_path=self.train_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )
        if self.val_path is not None:
            val_triplet_dataset = ContrastiveChannelMisinfoStanceDataset(
                pos_samples=self.pos_samples,
                neg_samples=self.neg_samples,
                data_path=self.val_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )
            val_infer_dataset = ContrastiveChannelMisinfoInferStanceDataset(
                pos_samples=1,
                neg_samples=1,
                data_path=self.val_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )
            self.val_dataset = [val_triplet_dataset, val_infer_dataset]
        if self.test_path is not None:
            test_triplet_dataset = ContrastiveChannelMisinfoStanceDataset(
                pos_samples=self.pos_samples,
                neg_samples=self.neg_samples,
                data_path=self.test_path,
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )
            test_infer_dataset = ContrastiveChannelMisinfoInferStanceDataset(
                pos_samples=1,
                neg_samples=1,
                data_path=[self.val_path, self.test_path],
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )

            self.test_dataset = [test_triplet_dataset, test_infer_dataset]
        if self.predict_path is not None:
            self.predict_dataset = ContrastiveChannelMisinfoInferStanceDataset(
                pos_samples=1,
                neg_samples=1,
                data_path=[self.val_path, self.predict_path],
                misinfo_path=self.misinfo_path,
                preprocess_config=preprocess_config,
            )

    def create_collator(self):
        return ContrastiveChannelBatchCollator(
            tokenizer=self.tokenizer,
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )


def tmp_stance(tmp_relation, tm_stance):
    # 0 (entail) + 1 = 1 % 2 = 1
    # 1 (contradict) + 1 = 2 % 2 = 0
    r_mod = (tmp_relation + 1) % 2
    # 1 % 2 = 1 + 1 = 2
    # 2 % 2 = 0 + 1 = 1
    # r_mod is 1 when entail, so flipping (tm_stance + 1) % 2 to get 1 is same as tm_stance
    # r_mod is 0 when contradict, so flipping tm_stance is correct
    return (tm_stance + r_mod) % 2 + 1


def flip_tm_stance(tm_stance):
    # 0 is no_stance
    # 1 is accept
    # 2 is reject
    # 0 -> 0
    # 1 -> 2
    # 2 -> 1
    # 0 % 2 = 0 + 0 = 0
    # 1 % 2 = 1 + 1 = 2
    # 2 % 2 = 0 + 1 = 1
    tm_flip_stance = (tm_stance % 2) + min(1, tm_stance)
    return tm_flip_stance


def flip_relation(tmp_relation):
    # 0 -> 1
    # 1 -> 0
    return (tmp_relation + 1) % 2
