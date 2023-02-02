import argparse
import os

import nltk
import numpy as np
import ujson as json
from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessingStopwords
from nltk.corpus import stopwords as stop_words
from tqdm.rich import tqdm


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                ex = json.loads(line)
                yield ex


def write_jsonl(path, data):
    with open(path, "w") as f:
        for ex in data:
            f.write(json.dumps(ex) + "\n")


def generate_preds(data_path, topics_predictions, total):
    for ex, pred in tqdm(zip(read_jsonl(data_path), topics_predictions), total=total):
        topic_number = int(np.argmax(pred))
        yield {"id": ex["id"], "topic": topic_number}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sentiment",
        type=str,
        default="positive",
        help="Sentiment to filter by",
        choices=["positive", "negative", "neutral"],
    )

    args = parser.parse_args()
    # Total: 5855583
    # neutral: 2419566
    # positive: 955470
    # negative: 2490010
    sentiment = args.sentiment

    sent_totals = {
        "neutral": 2419566,
        "positive": 955470,
        "negative": 2490010,
    }
    total = sent_totals[sentiment]

    data_folder = "/users/max/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt"
    data_path = os.path.join(data_folder, f"tweets-filtered-author-unique-reduced-sentiment-{sentiment}.jsonl")

    print("Loading data...")
    documents = [ex["text"].strip() for ex in tqdm(read_jsonl(data_path), total=total)]

    print("Preprocessing data...")
    nltk.download("stopwords")
    stopwords = list(stop_words.words("english"))
    sp = WhiteSpacePreprocessingStopwords(documents, stopwords_list=stopwords)
    preprocessed_documents, unpreprocessed_corpus, vocab, retained_indices = sp.preprocess()

    print("Creating TP...")
    tp = TopicModelDataPreparation("all-mpnet-base-v2")

    print("Training TP...")
    training_dataset = tp.fit(text_for_contextual=unpreprocessed_corpus, text_for_bow=preprocessed_documents)

    print("Training CTM...")
    ctm = CombinedTM(bow_size=len(tp.vocab), contextual_size=768, n_components=100, num_epochs=10)
    ctm.fit(training_dataset, n_samples=5)

    print("Saving topics...")
    topics = ctm.get_topic_lists(5)
    topics_path = os.path.join(data_folder, f"tweets-filtered-author-unique-reduced-sentiment-{sentiment}-topics.json")
    with open(topics_path, "w") as f:
        json.dump(topics, f)

    print("Saving predictions...")
    preds_path = os.path.join(
        data_folder, f"tweets-filtered-author-unique-reduced-sentiment-{sentiment}-topic-preds.jsonl"
    )
    write_jsonl(preds_path, generate_preds(data_path, ctm.training_doc_topic_distributions, total))

    print("Saving CTM...")
    model_path = os.path.join(data_folder, f"ctm-{sentiment}")
    os.makedirs(model_path, exist_ok=True)
    ctm.save(models_dir=model_path)

    print("DONE!")


if __name__ == "__main__":
    main()
