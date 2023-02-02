import os

import torch
import ujson as json
from tqdm.rich import tqdm
from transformers import pipeline


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


def preprocess(text):
    new_text = []
    for t in text.split(" "):
        t = "@user" if t.startswith("@") and len(t) > 1 else t
        t = "http" if t.startswith("http") else t
        new_text.append(t)
    return " ".join(new_text)


def generate(id_gen, text_gen, sentiment_task, total):
    for ex_id, out in tqdm(zip(id_gen, sentiment_task(text_gen, batch_size=128, num_workers=4)), total=total):
        pred = out["label"]
        yield {"id": ex_id, "pred": pred}


def main():
    data_folder = "/users/max/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt"
    data_path = os.path.join(data_folder, "tweets-filtered-author-unique-reduced.jsonl")
    output_path = os.path.join(data_folder, "tweets-filtered-author-unique-reduced-sentiment.jsonl")
    text_gen = (preprocess(ex["text"]) for ex in read_jsonl(data_path))
    id_gen = (ex["id"] for ex in read_jsonl(data_path))
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    # unique original tweets from ICWSM 2022 index from December 18th, 2019, and July 21st, 2021
    total = 5_865_046
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    sentiment_task = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name, device=device)
    # {'label': 'Negative', 'score': 0.7236}
    write_jsonl(output_path, generate(id_gen, text_gen, sentiment_task, total))
    print("DONE!")


if __name__ == "__main__":
    main()
