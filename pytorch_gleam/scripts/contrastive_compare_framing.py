import argparse
from collections import defaultdict

import ujson as json

from pytorch_gleam.data.twitter import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-f", "--frame_path", required=True)
    parser.add_argument("-p", "--predictions_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path
    frame_path = args.frame_path
    predictions_path = args.predictions_path
    output_path = args.output_path

    print("Loading data...")
    with open(frame_path) as f:
        frames = json.load(f)

    clusters = list(read_jsonl(input_path))
    c_lookup = {f["id"]: f for f in clusters}
    cluster_frame_preds = list(read_jsonl(predictions_path))

    c_scores = defaultdict(list)
    for prediction in cluster_frame_preds:
        c_id, f_id = prediction["ids"].split("|")
        c_f_score = prediction["scores"]
        c_scores[c_id].append((c_f_score, f_id))

    for c_id, c_sc in c_scores.items():
        c_f_score, c_f_id = max(c_sc, key=lambda x: x[0])
        c_f_txt = frames[c_f_id]["text"]
        c_lookup[c_id]["closest_frame"] = {c_f_id: {"score": c_f_score, "text": c_f_txt}}

    print("Saving results...")
    write_jsonl(output_path, clusters)

    print("Done!")


if __name__ == "__main__":
    main()
