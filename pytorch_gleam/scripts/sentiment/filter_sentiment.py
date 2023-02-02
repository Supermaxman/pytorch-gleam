import os

import ujson as json
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


def main():
    data_folder = "/users/max/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt"
    data_path = os.path.join(data_folder, "tweets-filtered-author-unique-reduced.jsonl")
    sent_path = os.path.join(data_folder, "tweets-filtered-author-unique-reduced-sentiment.jsonl")

    # Load sentiment data
    sent_data = {}
    for ex in read_jsonl(sent_path):
        sent_data[ex["id"]] = ex["pred"]

    sent_files = {
        "neutral": open(
            os.path.join(data_folder, "tweets-filtered-author-unique-reduced-sentiment-neutral.jsonl"), "w"
        ),
        "positive": open(
            os.path.join(data_folder, "tweets-filtered-author-unique-reduced-sentiment-positive.jsonl"), "w"
        ),
        "negative": open(
            os.path.join(data_folder, "tweets-filtered-author-unique-reduced-sentiment-negative.jsonl"), "w"
        ),
    }
    # unique original tweets from ICWSM 2022 index from December 18th, 2019, and July 21st, 2021
    total = 5_865_046
    try:
        for ex in tqdm(read_jsonl(data_path), total=total):
            ex_id = ex["id"]
            sent = sent_data[ex_id]
            f = sent_files[sent]
            f.write(json.dumps({"id": ex_id, "text": ex["text"]}) + "\n")
    finally:
        for f in sent_files.values():
            f.close()

    print("DONE!")


if __name__ == "__main__":
    main()
