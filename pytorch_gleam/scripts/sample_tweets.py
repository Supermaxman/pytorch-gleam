import argparse
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
from tqdm import tqdm

from pytorch_gleam.data.twitter import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-n", "--num_examples", type=int, required=True)
    parser.add_argument("-q", "--num_queries", type=int, required=True)
    parser.add_argument("-t", "--num_samples", type=int, required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    seed = args.seed
    n = args.num_examples
    q = args.num_queries
    t = args.num_samples

    pl.seed_everything(seed)

    samples = np.random.randint(low=0, high=n, size=[q, t])

    q_samples = {f'RQ{q_idx+1}': set([int(x) for x in samples[q_idx].tolist()]) for q_idx in range(q)}
    outputs = []
    for t_idx, tweet in enumerate(tqdm(read_jsonl(input_path), total=n)):
        added = False
        tweet['candidates'] = {}
        for q_id, q_s in q_samples.items():
            if t_idx in q_s:
                added = True
                tweet['candidates'][q_id] = {}

        if added:
            outputs.append(tweet)

    write_jsonl(output_path, outputs)


if __name__ == "__main__":
    main()
