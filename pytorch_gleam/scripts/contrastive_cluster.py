import argparse
import itertools
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

from pytorch_gleam.data.twitter import read_jsonl, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-p", "--predictions_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-t", "--threshold", type=float, default=4.0)
    parser.add_argument("-c", "--clustering", default="complete")
    args = parser.parse_args()

    threshold = args.threshold
    clustering = args.clustering
    input_path = args.input_path
    predictions_path = args.predictions_path
    output_path = args.output_path
    seed = args.seed

    pl.seed_everything(seed)

    data = list(read_jsonl(input_path))
    predictions = list(read_jsonl(predictions_path))
    d = np.array([-x["scores"] for x in predictions])
    d_min = np.min(d)
    dists = {tuple(p["ids"].split("|")): -p["scores"] - d_min for p in predictions}

    q_examples = defaultdict(list)
    examples = {ex["id"]: ex for ex in data}
    for ex in data:
        for q_id, q in ex["candidates"].items():
            q_examples[q_id].append(ex)

    outputs = []
    for q_id, q_exs in tqdm(q_examples.items()):
        q_ex_idx = {}
        n = len(q_exs)
        q_dists = np.zeros(shape=[n, n], dtype=np.float32)
        for a_ex, b_ex in itertools.combinations(q_exs, 2):
            a_id = a_ex["id"]
            if a_id not in q_ex_idx:
                q_ex_idx[a_id] = len(q_ex_idx)
            b_id = b_ex["id"]
            if b_id not in q_ex_idx:
                q_ex_idx[b_id] = len(q_ex_idx)
            a_b_key = a_id, b_id
            b_a_key = b_id, a_id
            if a_b_key in dists:
                d = dists[a_b_key]
            else:
                d = dists[b_a_key]
            q_dists[q_ex_idx[a_id], q_ex_idx[b_id]] = d
            q_dists[q_ex_idx[b_id], q_ex_idx[a_id]] = d

        q_idx_ex = {v: k for k, v in q_ex_idx.items()}
        cluster = AgglomerativeClustering(
            n_clusters=None,
            affinity="precomputed",
            linkage=clustering,
            distance_threshold=threshold,
        ).fit(q_dists)

        labels = cluster.labels_

        clusters = defaultdict(list)
        for idx, c_id in enumerate(labels):
            ex_id = q_idx_ex[idx]
            ex = examples[ex_id]
            clusters[c_id].append(ex)

        sorted_clusters = {}
        for c_id, c_exs in clusters.items():
            if len(c_exs) == 1:
                sorted_clusters[c_id] = c_exs
                continue

            ex_dists = defaultdict(list)
            for a_ex, b_ex in itertools.combinations(c_exs, 2):
                a_id = a_ex["id"]
                b_id = b_ex["id"]
                a_b_key = a_id, b_id
                b_a_key = b_id, a_id
                if a_b_key in dists:
                    d = dists[a_b_key]
                else:
                    d = dists[b_a_key]
                ex_dists[a_id].append(d)
                ex_dists[b_id].append(d)

            ex_ordering = {}
            for ex_id, ex_ds in ex_dists.items():
                ex_avg_d = np.mean(ex_ds)
                ex_ordering[ex_id] = float(ex_avg_d)
            # sort lowest to highest average distance to other examples in cluster
            sorted_clusters[c_id] = sorted(c_exs, key=lambda ex: ex_ordering[ex["id"]])

        clusters = {
            f"{q_id}-C{k}": v
            for k, v in sorted(sorted_clusters.items(), key=lambda x: len(x[1]), reverse=True)
            if len(v) > 1
        }
        c_list = [{"id": k, "docs": v} for k, v in clusters.items()]
        outputs.extend(c_list)

    write_jsonl(output_path, outputs)


if __name__ == "__main__":
    main()
