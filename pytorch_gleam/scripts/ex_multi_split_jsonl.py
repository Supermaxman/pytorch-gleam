import argparse
import contextlib
import itertools
import math
import os
import random

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
    parser.add_argument("-i", "--input_paths", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-a", "--output_a_name", default="train")
    parser.add_argument("-b", "--output_b_name", default="test")
    parser.add_argument("-ka", "--output_a_files", type=int, default=10)
    parser.add_argument("-kb", "--output_b_files", type=int, default=1)
    parser.add_argument("-n", "--number_of_examples", type=int)
    parser.add_argument("-r", "--ratio", type=float, default=0.01)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    # csv
    input_paths = args.input_paths
    output_path = args.output_path
    n = args.number_of_examples
    num_a_files = args.output_a_files
    num_b_files = args.output_b_files
    output_a_name = args.output_a_name
    output_b_name = args.output_b_name
    ratio = args.ratio
    seed = args.seed

    random.seed(seed)

    assert 0.0 < ratio < 1.0
    assert output_a_name != output_b_name

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    output_a_path = os.path.join(output_path, output_a_name)
    output_b_path = os.path.join(output_path, output_b_name)

    assert num_a_files > 0
    assert num_b_files > 0

    full_indices = list(range(n))
    random.shuffle(full_indices)
    b_size = int(math.ceil(ratio * n))
    a_size = n - b_size

    a_indices, b_indices = set(full_indices[:a_size]), set(full_indices[a_size:])

    assert len(a_indices.intersection(b_indices)) == 0

    ex_generator = itertools.chain.from_iterable(read_jsonl(path) for path in input_paths.split(","))
    with contextlib.ExitStack() as stack:
        a_files = [stack.enter_context(open(output_a_path + f"-{i}.jsonl", "w")) for i in range(num_a_files)]
        b_files = [stack.enter_context(open(output_b_path + f"-{i}.jsonl", "w")) for i in range(num_b_files)]
        for i, ex in tqdm(enumerate(ex_generator), total=n):
            # only keep id and text for this splitting step
            ex_jsonl = json.dumps({"id": ex["id"], "text": ex["text"]}) + "\n"
            if i in a_indices:
                output_file = a_files[i % num_a_files]
            elif i in b_indices:
                output_file = b_files[i % num_b_files]
            else:
                raise ValueError(f"Index {i} not in either split!")

            output_file.write(ex_jsonl)


if __name__ == "__main__":
    main()
