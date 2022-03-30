import argparse
import random
from multiprocessing import Pool

import pytorch_lightning as pl
import ujson as json
from spacy.lang.en import English
from spacy.language import Language
from tqdm import tqdm
from transformers import AutoTokenizer, BertTokenizer

from pytorch_gleam.data.datasets import BertPreTrainDataConfig
from pytorch_gleam.data.twitter import preprocess_tweet, read_jsonl, TweetPreprocessConfig


@Language.component("avoid_sentencizer_hashtags")
def _avoid_sentence_boundary_on_hashtag(doc):
    for token in doc[:-1]:
        if token.text == "#":
            doc[token.i + 1].is_sent_start = False
    return doc


nlp = English()
# sentencizer = nlp.create_pipe("sentencizer")
# nlp.add_pipe(sentencizer)
nlp.add_pipe("sentencizer")
nlp.add_pipe("avoid_sentencizer_hashtags")
tokenizer: BertTokenizer = None
data_config: BertPreTrainDataConfig = None
tokenizer_config: TweetPreprocessConfig = None
vocab_words: list = []


def create_example(instance):
    # tokens is a list of token strings, needs to be converted to ids
    # segment_lengths is list of ints of lengths of token_type_ids
    # is_random_next is bool label for seq pred task
    # masked_lm_positions is indices of [mask]
    # masked_lm_labels is token strings of [mask] indices
    input_ids = tokenizer.convert_tokens_to_ids(instance["tokens"])
    masked_lm_positions = list(instance["masked_lm_positions"])
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance["masked_lm_labels"])
    next_sentence_label = 1 if instance["is_random_next"] else 0
    example = {
        "input_ids": input_ids,
        # "attention_mask": [1] * len(input_ids),
        "segment_lengths": instance["segment_lengths"],
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_ids": masked_lm_ids,
        "next_sentence_label": next_sentence_label,
    }

    return example


def read_text(data_path):
    for ex in read_jsonl(data_path):
        ex_text = ex["text"]
        ex_text = preprocess_tweet(ex_text, tokenizer_config)
        yield ex_text


def read_path(data_path, n):
    documents = []
    for text in tqdm(read_text(data_path), leave=True, total=n):
        document = []
        doc = nlp(text)
        for s_idx, sent in enumerate(doc.sents):
            tokens = tokenizer.tokenize(sent.text, add_special_tokens=False)
            # min token requirement
            if len(tokens) < 4:
                continue
            document.append(tokens)
        # non-empty documents
        if len(document) == 0:
            continue
        documents.append(document)
    return documents


def create_instances_from_document(document, sample_docs):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = data_config.max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random.random() < data_config.short_seq_prob:
        # target_seq_length = random.randint(2, max_num_tokens)
        target_seq_length = random.randint(max(max_num_tokens // 4, 4), max_num_tokens)

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
                    a_end = random.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []
                # Random next
                is_random_next = False
                if len(current_chunk) == 1 or random.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    random_document_index = random.randint(0, len(sample_docs) - 1)
                    random_document = sample_docs[random_document_index]
                    random_start = random.randint(0, len(random_document) - 1)
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
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens)

                # this could lead to an infinite loop
                # if it got so truncated or was so small then skip
                # if len(tokens_a) < 1 or len(tokens_b) < 1:
                #     continue
                # make sure tokens for sentences are non-zero
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_lengths = []
                tokens.append("[CLS]")
                for token in tokens_a:
                    tokens.append(token)

                tokens.append("[SEP]")
                # [CLS] + tokens_a + [SEP] is seq 0
                segment_lengths.append(len(tokens_a) + 2)

                for token in tokens_b:
                    tokens.append(token)
                tokens.append("[SEP]")
                # tokens_b + [SEP] is seq 1
                segment_lengths.append(len(tokens_b) + 1)

                tokens, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(
                    tokens,
                    data_config.masked_lm_prob,
                    data_config.max_predictions_per_seq,
                    data_config.do_whole_word_mask,
                )
                instance = {
                    "tokens": tokens,
                    "segment_lengths": segment_lengths,
                    "is_random_next": is_random_next,
                    "masked_lm_positions": masked_lm_positions,
                    "masked_lm_labels": masked_lm_labels,
                }
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    # tokens is a list of token strings, needs to be converted to ids
    # segment_lengths is list of lengths of token_type_ids
    # is_random_next is bool label for seq pred task
    # masked_lm_positions is indices of [mask]
    # masked_lm_labels is token strings of [mask] indices
    return instances


def create_masked_lm_predictions(
    tokens,
    masked_lm_prob,
    max_predictions_per_seq,
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

    random.shuffle(cand_indexes)

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
            if random.random() < 0.8:
                masked_token = "[MASK]"
            else:
                # 10% of the time, keep original
                if random.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[random.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token
            lm_instance = {"index": index, "label": tokens[index]}
            masked_lms.append(lm_instance)
    # assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x["index"])

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p["index"])
        masked_lm_labels.append(p["label"])

    return output_tokens, masked_lm_positions, masked_lm_labels


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def process_text(text):
    document = []
    doc = nlp(text)
    for s_idx, sent in enumerate(doc.sents):
        tokens = tokenizer.tokenize(sent.text, add_special_tokens=False)
        # min token requirement
        if len(tokens) < 4:
            continue
        document.append(tokens)
    # non-empty documents
    if len(document) == 0:
        return None
    return document


def sample_documents(doc_idx, all_docs, num_samples):
    samples = []
    for _ in range(num_samples):
        random_document_index = doc_idx
        while random_document_index == doc_idx:
            random_document_index = random.randint(0, len(all_docs) - 1)
        random_document = all_docs[random_document_index]
        samples.append(random_document)
    return samples


def create_instances(args):
    doc, sample_docs = args
    examples = []
    for _ in range(data_config.dupe_factor):
        for instance in create_instances_from_document(doc, sample_docs):
            example = create_example(instance)
            examples.append(example)
    return examples


def doc_sample_iterator(documents):
    for d_index, doc in enumerate(documents):
        sample_docs = sample_documents(d_index, documents, num_samples=20)
        yield doc, sample_docs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-n", "--number_of_examples", type=int, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    n = args.number_of_examples
    seed = args.seed

    pl.seed_everything(seed)

    global tokenizer_config
    global data_config
    global tokenizer
    global vocab_words

    # max_seq_len: 128
    # masked_lm_prob: 0.15
    # short_seq_prob: 0.10
    # max_predictions_per_seq: 20
    # dupe_factor: 10
    tokenizer_name = "digitalepidemiologylab/covid-twitter-bert-v2"
    tokenizer_config = TweetPreprocessConfig()
    data_config = BertPreTrainDataConfig()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    vocab_words = list(tokenizer.vocab.keys())

    print("Processing documents...")
    documents = []
    with Pool(processes=8) as p:
        for doc in tqdm(p.imap_unordered(process_text, read_text(input_path)), total=n):
            if doc is None:
                continue
            documents.append(doc)

    num_examples = data_config.dupe_factor * len(documents)

    print(f"Processed documents: {num_examples:,}")

    print("Processing instances...")
    instance_count = 0
    with open(output_path, "w") as f:
        with Pool(processes=8) as p:
            for instances in tqdm(
                p.imap_unordered(create_instances, doc_sample_iterator(documents)), total=num_examples
            ):
                for instance in instances:
                    f.write(json.dumps(instance) + "\n")
                    instance_count += 1

    print(f"Processed instances: {instance_count:,}")


if __name__ == "__main__":
    main()
