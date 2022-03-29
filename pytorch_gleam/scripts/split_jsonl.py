import argparse
import itertools
import math
import os.path
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
    parser.add_argument("-a", "--output_a_name", default="train.jsonl")
    parser.add_argument("-b", "--output_b_name", default="test.jsonl")
    parser.add_argument("-n", "--number_of_examples", type=int)
    parser.add_argument("-r", "--ratio", type=float, default=0.2)
    parser.add_argument("-s", "--seed", type=int, default=0)
    args = parser.parse_args()

    # csv
    input_paths = args.input_paths
    output_path = args.output_path
    output_a_name = args.output_a_name
    output_b_name = args.output_b_name
    n = args.number_of_examples
    ratio = args.ratio
    seed = args.seed

    random.seed(seed)

    assert 0.0 < ratio < 1.0
    assert output_a_name != output_b_name

    output_a_path = os.path.join(output_path, output_a_name)
    output_b_path = os.path.join(output_path, output_b_name)

    full_indices = list(range(n))
    random.shuffle(full_indices)
    b_size = int(math.ceil(ratio * n))
    a_size = n - b_size

    a_indices, b_indices = set(full_indices[:a_size]), set(full_indices[a_size:])

    assert len(a_indices.intersection(b_indices)) == 0

    ex_generator = itertools.chain.from_iterable(read_jsonl(path) for path in input_paths.split(","))
    with open(output_a_path, "w") as a_file, open(output_b_path, "w") as b_file:
        for i, ex in tqdm(enumerate(ex_generator), total=n):
            ex_jsonl = json.dumps(ex) + "\n"
            if i in a_indices:
                a_file.write(ex_jsonl)
            elif i in b_indices:
                b_file.write(ex_jsonl)
            else:
                raise ValueError(f"Index {i} not in either split!")


if __name__ == "__main__":
    main()
