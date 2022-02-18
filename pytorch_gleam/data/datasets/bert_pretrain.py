import dataclasses
import json
import random
from typing import List, Union
import re
import unicodedata
import html
import emoji
import unidecode

from spacy.lang.en import English
from spacy.language import Language

import torch
from torch.utils.data import Dataset
from pytorch_gleam.data.datasets.base_datasets import BaseDataModule
from pytorch_gleam.data.collators import BertPreBatchCollator


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


nlp = build_spacy_model()
# compile regexes
username_regex = re.compile(r"(^|[^@\w])@(\w{1,15})\b")
url_regex = re.compile(r"((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))")
control_char_regex = re.compile(r"[\r\n\t]+")
# translate table for punctuation
transl_table = dict([(ord(x), ord(y)) for x, y in zip("‘’´“”–-", "'''\"\"--")])


@dataclasses.dataclass
class BertTokenizerConfig:
    do_lower_case: bool = True
    url_filler: str = "twitterurl"
    username_filler: str = "twitteruser"
    replace_usernames: bool = True
    replace_urls: bool = True
    asciify_emojis: bool = True
    replace_multiple_usernames: bool = True
    replace_multiple_urls: bool = True
    standardize_punctuation: bool = True
    remove_unicode_symbols: bool = True
    remove_accented_characters: bool = False


def preprocess_bert(text: str, args: BertTokenizerConfig):
    """Preprocesses tweet for BERT"""
    # standardize
    text = standardize_text(text)
    # replace usernames/urls
    if args.replace_usernames:
        text = replace_usernames(text, filler=args.username_filler)
    if args.replace_urls:
        text = replace_urls(text, filler=args.url_filler)
    if args.asciify_emojis:
        text = asciify_emojis(text)
    if args.standardize_punctuation:
        text = standardize_punctuation(text)
    if args.do_lower_case:
        text = text.lower()
    if args.replace_multiple_usernames:
        text = replace_multi_occurrences(text, args.username_filler)
    if args.replace_multiple_urls:
        text = replace_multi_occurrences(text, args.url_filler)
    if args.remove_unicode_symbols:
        text = remove_unicode_symbols(text)
    if args.remove_accented_characters:
        text = remove_accented_characters(text)
    return text


def remove_accented_characters(text):
    text = unidecode.unidecode(text)
    return text


def remove_unicode_symbols(text):
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "So")
    return text


def replace_multi_occurrences(text, filler):
    """Replaces multiple occurrences of filler with n filler"""
    # only run if we have multiple occurrences of filler
    if text.count(filler) <= 1:
        return text
    # pad fillers with whitespace
    text = text.replace(f"{filler}", f" {filler} ")
    # remove introduced duplicate whitespaces
    text = " ".join(text.split())
    # find indices of occurrences
    indices = []
    for m in re.finditer(r"{}".format(filler), text):
        index = m.start()
        indices.append(index)
    # collect merge list
    merge_list = []
    old_index = None
    for i, index in enumerate(indices):
        if i > 0 and index - old_index == len(filler) + 1:
            # found two consecutive fillers
            if len(merge_list) > 0 and merge_list[-1][1] == old_index:
                # extend previous item
                merge_list[-1][1] = index
                merge_list[-1][2] += 1
            else:
                # create new item
                merge_list.append([old_index, index, 2])
        old_index = index
    # merge occurrences
    if len(merge_list) > 0:
        new_text = ""
        pos = 0
        for (start, end, count) in merge_list:
            new_text += text[pos:start]
            new_text += f"{count} {filler}"
            pos = end + len(filler)
        new_text += text[pos:]
        text = new_text
    return text


def asciify_emojis(text):
    """
    Converts emojis into text aliases. E.g. 👍 becomes :thumbs_up:
    For a full list of text aliases see: https://www.webfx.com/tools/emoji-cheat-sheet/
    """
    text = emoji.demojize(text)
    return text


def standardize_text(text):
    """
    1) Escape HTML
    2) Replaces some non-standard punctuation with standard versions.
    3) Replace \r, \n and \t with white spaces
    4) Removes all other control characters and the NULL byte
    5) Removes duplicate white spaces
    """
    # escape HTML symbols
    text = html.unescape(text)
    # standardize punctuation
    text = text.translate(transl_table)
    text = text.replace("…", "...")
    # replace \t, \n and \r characters by a whitespace
    text = re.sub(control_char_regex, " ", text)
    # remove all remaining control characters
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C")
    # replace multiple spaces with single space
    text = " ".join(text.split())
    return text.strip()


def standardize_punctuation(text):
    return "".join(
        [
            unidecode.unidecode(t) if unicodedata.category(t)[0] == "P" else t
            for t in text
        ]
    )


def replace_usernames(text, filler="user"):
    # @<user> is a marker used internally. use filler instead
    text = text.replace("@<user>", f"{filler}")
    # replace other user handles by filler
    text = re.sub(username_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f" {filler} ")
    text = " ".join(text.split())
    return text


def replace_urls(text, filler="url"):
    # <url> is a marker used internally. use filler instead
    text = text.replace("<url>", filler)
    # replace other urls by filler
    text = re.sub(url_regex, filler, text)
    # add spaces between, and remove double spaces again
    text = text.replace(filler, f" {filler} ")
    text = " ".join(text.split())
    return text


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

                (
                    tokens,
                    masked_lm_positions,
                    masked_lm_labels,
                ) = create_masked_lm_predictions(
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

    num_to_predict = min(
        max_predictions_per_seq, max(1, int(round(len(tokens) * masked_lm_prob)))
    )

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


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


class BertPreDataset(Dataset):
    def __init__(
        self,
        tokenizer_config: BertTokenizerConfig,
        masked_lm_prob: float,
        data_path: Union[str, List[str]],
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
        self.vocab_words = list(tokenizer.vocab.keys())

        if isinstance(data_path, str):
            self.read_path(data_path)
        else:
            for stage, stage_path in enumerate(data_path):
                self.read_path(stage_path, stage)

        self.rng = random.Random(torch.seed())
        self.rng.shuffle(self.documents)
        self.create_examples()
        self.documents = None
        self.rng.shuffle(self.examples)

        self.num_examples = len(self.examples)

    def create_examples(self):
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

    def create_example(self, instance):
        # tokens is a list of token strings, needs to be converted to ids
        # segment_ids is list of ints of token_type_ids
        # is_random_next is bool label for seq pred task
        # masked_lm_positions is indices of [mask]
        # masked_lm_labels is token strings of [mask] indices
        input_ids = self.tokenizer.convert_tokens_to_ids(instance["tokens"])
        masked_lm_positions = list(instance["masked_lm_positions"])
        masked_lm_ids = self.tokenizer.convert_tokens_to_ids(
            instance["masked_lm_labels"]
        )
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
            ex_text = preprocess_bert(ex_text, self.tokenizer_config)
            doc = nlp(ex_text)
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
        bert_tokenizer_config: BertTokenizerConfig = None,
        masked_lm_prob: float = 0.15,
        short_seq_prob: float = 0.10,
        max_predictions_per_seq: int = 14,
        dupe_factor: int = 10,
        do_whole_word_mask: bool = True,
        train_path: Union[str, List[str]] = None,
        val_path: Union[str, List[str]] = None,
        test_path: Union[str, List[str]] = None,
        predict_path: Union[str, List[str]] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if bert_tokenizer_config is None:
            bert_tokenizer_config = BertTokenizerConfig()
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

        if self.train_path is not None:
            self.train_dataset = BertPreDataset(
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                data_path=self.train_path,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
            )
        if self.val_path is not None:
            self.val_dataset = BertPreDataset(
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                data_path=self.val_path,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
            )
        if self.test_path is not None:
            self.test_dataset = BertPreDataset(
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                data_path=self.test_path,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
            )
        if self.predict_path is not None:
            self.predict_dataset = BertPreDataset(
                tokenizer_config=self.bert_tokenizer_config,
                masked_lm_prob=self.masked_lm_prob,
                tokenizer=self.tokenizer,
                data_path=self.predict_path,
                short_seq_prob=self.short_seq_prob,
                max_seq_length=self.max_seq_len,
                max_predictions_per_seq=self.max_predictions_per_seq,
                dupe_factor=self.dupe_factor,
                do_whole_word_mask=self.do_whole_word_mask,
            )

    def create_collator(self):
        return BertPreBatchCollator(
            max_seq_len=self.max_seq_len,
            max_predictions_per_seq=self.max_predictions_per_seq,
            use_tpus=self.use_tpus,
        )
