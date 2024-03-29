import argparse
import json
import os
from collections import defaultdict

import torch


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


def load_predictions(input_path):
    pred_list = []
    for file_name in os.listdir(input_path):
        if file_name.endswith(".pt"):
            preds = torch.load(os.path.join(input_path, file_name))
            pred_list.extend(preds)
        if file_name.endswith(".jsonl"):
            preds = read_jsonl(os.path.join(input_path, file_name))
            pred_list.extend(preds)

    question_scores = defaultdict(lambda: defaultdict(dict))
    p_count = 0
    u_count = 0
    for prediction in pred_list:
        doc_pass_id = prediction["id"]
        q_p_id = prediction["question_id"]
        # score = prediction['pos_score']
        score = prediction["pos_score"] - prediction["neg_score"]
        if doc_pass_id not in question_scores or q_p_id not in question_scores[doc_pass_id]:
            p_count += 1
        u_count += 1
        question_scores[doc_pass_id][q_p_id] = score
    print(f"{p_count} unique predictions")
    print(f"{u_count} total predictions")
    return question_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path

    question_scores = load_predictions(input_path)
    with open(output_path, "w") as f:
        json.dump(question_scores, f)


if __name__ == "__main__":
    main()
