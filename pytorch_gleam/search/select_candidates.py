import argparse
import os
from collections import defaultdict

import ujson as json
from tqdm import tqdm


def read_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    ex = json.loads(line)
                    yield ex
                except Exception as e:
                    print(e)


def write_jsonl(data, path):
    with open(path, "w") as f:
        for example in data:
            json_data = json.dumps(example)
            f.write(json_data + "\n")


def get_posts(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        for ex in read_jsonl(file_path):
            yield ex


def collect_posts(dir_path, post_candidates):
    for post in tqdm(get_posts(dir_path), total=46_159_226):
        post_id = post["id"]
        if post_id not in post_candidates:
            continue
        post["candidates"] = post_candidates[post_id]
        yield post


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--data_path", required=True)
    parser.add_argument("-sc", "--scores_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-mis", "--min_score", default=0.0, type=float)
    args = parser.parse_args()

    print("Loading scores...")
    question_scores = defaultdict(list)
    for file in tqdm(sorted(os.listdir(args.scores_path), key=lambda x: int(x.split("-")[-1].split(".")[0]))):
        for ex in read_jsonl(os.path.join(args.scores_path, file)):
            score = ex["pos_score"] - ex["neg_score"]
            if score > args.min_score:
                q_id = ex["question_id"]
                post_id = ex["id"]
                question_scores[q_id].append((score, post_id))

    print("Sorting posts for each subquestion...")
    for q_id in tqdm(list(question_scores)):
        question_scores[q_id] = sorted(
            question_scores[q_id],
            # (score, post_id)
            key=lambda x: x[0],
            reverse=True,
        )

    print("Collecting candidates for each post...")
    post_candidates = defaultdict(dict)
    for q_id, q_rel in tqdm(question_scores.items()):
        for rank, (t_score, post_id) in enumerate(q_rel, start=1):
            p_candidates = post_candidates[post_id]
            p_candidates[q_id] = {"rank": rank, "score": t_score}

    print("Writing candidate posts...")
    write_jsonl(collect_posts(args.data_path, post_candidates), args.output_path)


if __name__ == "__main__":
    main()
