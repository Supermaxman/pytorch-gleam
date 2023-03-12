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
                ex = json.loads(line)
                yield ex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-f", "--tweet_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-t", "--threshold", default=0.3, type=float)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    tweet_path = args.tweet_path
    threshold = args.threshold
    count_lookup = {}
    time_lookup = {}

    print("building tweet-info index")
    for tweet in tqdm(read_jsonl(tweet_path), total=46_159_226):
        count_lookup[tweet["id"]] = 1 + tweet["public_metrics"]["retweet_count"]
        time_lookup[tweet["id"]] = tweet["created_at"]

    print("loading scores")
    frame_stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for file in tqdm(sorted(os.listdir(input_path), key=lambda x: int(x.split("-")[-1].split(".")[0]))):
        for pred in tqdm(read_jsonl(os.path.join(input_path, file)), total=36_668_907):
            tweet_id, f_id = pred["ids"].split("|")
            _, accept_score, reject_score = pred["scores"]
            ts = time_lookup[tweet_id]
            count = count_lookup[tweet_id]
            # parse timestamp as datetime and convert to YYYY-MM-DD
            # 2020-11-09T21:32:22.000Z
            day = ts.split("T")[0]
            if accept_score > threshold or reject_score > threshold:
                stance = "Accept"
                if accept_score < reject_score:
                    stance = "Reject"
                frame_stats[f_id][stance][day] += count

    print("saving results")
    with open(output_path, "w") as f:
        for f_id, stance_stats in tqdm(frame_stats.items()):
            for stance, day_stats in stance_stats.items():
                for day, count in day_stats.items():
                    f.write(json.dumps({"f_id": f_id, "Date": day, "Stance": stance, "tweets": count}) + "\n")


if __name__ == "__main__":
    main()
