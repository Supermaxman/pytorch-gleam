import random
from typing import List, Union

import torch
from spacy.lang.en import English
from spacy.language import Language
from torch.utils.data import Dataset
from tqdm import tqdm

from pytorch_gleam.data.collators import BertPreBatchCollator
from pytorch_gleam.data.datasets import preprocess_tweet, read_jsonl, TweetPreprocessConfig
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule


@Language.component("avoid_sentencizer_hashtags")
def _avoid_sentence_boundary_on_hashtag(doc):
    for token in doc[:-1]:
        if token.text == "#":
            doc[token.i + 1].is_sent_start = False
    return doc


# build spacy model
def build_spacy_model():
    nlp = English()
    # sentencizer = nlp.create_pipe("sentencizer")
    # nlp.add_pipe(sentencizer)
    nlp.add_pipe("sentencizer")
    nlp.add_pipe("avoid_sentencizer_hashtags")
    return nlp


class BertPreDataset(Dataset):
    def __init__(
        self,
        tokenizer_config: TweetPreprocessConfig,
        masked_lm_prob: float,
        tokenizer,
        short_seq_prob: float,
        max_seq_length: int,
        max_predictions_per_seq: int,
        dupe_factor: int,
        do_whole_word_mask: bool,
    ):
        super().__init__()
        self.short_seq_prob = short_seq_prob
        self.max_seq_length = max_seq_length
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor
        self.do_whole_word_mask = do_whole_word_mask
        self.masked_lm_prob = masked_lm_prob
        self.tokenizer_config = tokenizer_config
        self.tokenizer = tokenizer
        self.documents = []
        self.examples = []
        self.vocab_words = None
        self.nlp = None
        self.rng = None
        self.num_examples = 0

    def load(self, data_path):
        self.rng = random.Random(torch.seed())
        self.nlp = build_spacy_model()
        self.vocab_words = list(self.tokenizer.vocab.keys())

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

        self.rng.shuffle(self.documents)
        self.create_examples()
        self.documents = None
        self.rng.shuffle(self.examples)
        self.num_examples = len(self.examples)
        self.nlp = None
        self.rng = None
        self.vocab_words = None

    def create_examples(self):
        with tqdm(total=self.dupe_factor * len(self.documents)) as progress:
            for _ in range(self.dupe_factor):
                for document_index in range(len(self.documents)):
                    for instance in create_instances_from_document(
                        self.documents,
                        document_index,
                        self.max_seq_length,
                        self.short_seq_prob,
                        self.masked_lm_prob,
                        self.max_predictions_per_seq,
                        self.vocab_words,
                        self.rng,
                        self.do_whole_word_mask,
                    ):
                        example = self.create_example(instance)
                        self.examples.append(example)
                    progress.update(1)

    def create_example(self, instance):
        # tokens is a list of token strings, needs to be converted to ids
        # segment_ids is list of ints of token_type_ids
        # is_random_next is bool label for seq pred task
        # masked_lm_positions is indices of [mask]
        # masked_lm_labels is token strings of [mask] indices
        input_ids = self.tokenizer.convert_tokens_to_ids(instance["tokens"])
        masked_lm_positions = list(instance["masked_lm_positions"])
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(instance["masked_lm_labels"])
        masked_lm_weights = [1.0] * len(masked_lm_ids)
        next_sentence_label = 1 if instance["is_random_next"] else 0
        example = {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": list(instance["segment_ids"]),
            "masked_lm_positions": masked_lm_positions,
            "masked_lm_ids": masked_lm_ids,
            "masked_lm_weights": masked_lm_weights,
            "next_sentence_label": next_sentence_label,
        }

        return example

    def read_path(self, data_path, stage=0):
        for ex in read_jsonl(data_path):
            ex_text = ex["full_text"] if "full_text" in ex else ex["text"]
            ex_text = preprocess_tweet(ex_text, self.tokenizer_config)
            doc = self.nlp(ex_text)
            document = []
            for s_idx, sent in enumerate(doc.sents):
                tokens = self.tokenizer.tokenize(sent.text, add_special_tokens=False)
                document.append(tokens)

            self.documents.append(document)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        example = self.examples[idx]

        return example

    def worker_init_fn(self, _):
        pass


class BertPreDataModule(BaseDataModule):
    def __init__(
        self,
        bert_tokenizer_config: TweetPreprocessConfig = None,
        masked_lm_prob: float = 0.15,
        short_seq_prob: float = 0.10,
        max_predictions_per_seq: int = 14,
        dupe_factor: int = 10,
        do_whole_word_mask: bool = True,
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        test_path: Union[str, List[str]] = None,
        predict_path: Union[str, List[str]] = None,
        pickle_path: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bert_tokenizer_config is None:
            bert_tokenizer_config = TweetPreprocessConfig()
        self.short_seq_prob = short_seq_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.dupe_factor = dupe_factor
        self.do_whole_word_mask = do_whole_word_mask

        self.bert_tokenizer_config = bert_tokenizer_config
        self.masked_lm_prob = masked_lm_prob
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.pickle_path = pickle_path

        if self.train_path is not None:
            self.train_dataset = self.load_or_create(
                BertPreDataset,
                self.train_path,
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
                pickle_path=self.pickle_path,
            )
        if self.val_path is not None:
            self.val_dataset = self.load_or_create(
                BertPreDataset,
                self.val_path,
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
                pickle_path=self.pickle_path,
            )
        if self.test_path is not None:
            self.test_dataset = self.load_or_create(
                BertPreDataset,
                self.test_path,
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
                pickle_path=self.pickle_path,
            )
        if self.predict_path is not None:
            self.predict_dataset = self.load_or_create(
                BertPreDataset,
                self.predict_path,
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
                pickle_path=self.pickle_path,
            )

    def create_collator(self):
        return BertPreBatchCollator(
            max_seq_len=self.max_seq_len,
            use_tpus=self.use_tpus,
        )


def create_instances_from_document(
    all_documents,
    document_index,
    max_seq_length,
    short_seq_prob,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
    do_whole_word_mask,
):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    random_document_index = None
                    for _ in range(10):
                        random_document_index = rng.randint(0, len(all_documents) - 1)
                        if random_document_index != document_index:
                            break

                    random_document = all_documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions, masked_lm_labels,) = create_masked_lm_predictions(
                    tokens,
                    masked_lm_prob,
                    max_predictions_per_seq,
                    vocab_words,
                    rng,
                    do_whole_word_mask,
                )
                instance = {
                    "tokens": tokens,
                    "segment_ids": segment_ids,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    # tokens is a list of token strings, needs to be converted to ids
    # segment_ids is list of ints of token_type_ids
    # is_random_next is bool label for seq pred task
    # masked_lm_positions is indices of [mask]
    # masked_lm_labels is token strings of [mask] indices
    return instances


def create_masked_lm_predictions(
    tokens,
    masked_lm_prob,
    max_predictions_per_seq,
    vocab_words,
    rng,
    do_whole_word_mask,
):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        # Whole Word Masking means that if we mask all of the wordpieces
        # corresponding to an original word. When a word has been split into
        # WordPieces, the first token does not have any marker and any subsequence
        # tokens are prefixed with ##. So whenever we see the ## token, we
        # append it to the previous set of word indexes.
        #
        # Note that Whole Word Masking does *not* change the training code
        # at all -- we still predict each WordPiece independently, softmaxed
        # over the entire vocabulary.
        if do_whole_word_mask and len(cand_indexes) >= 1 and token.startswith("##"):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index_set in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token
            lm_instance = {"index": index, "label": tokens[index]}
            masked_lms.append(lm_instance)
    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x["index"])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p["index"])
        masked_lm_labels.append(p["label"])

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()
