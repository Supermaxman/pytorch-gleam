import argparse
import itertools
import os

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
    parser.add_argument("-n", "--number_of_examples", type=int)
    parser.add_argument("-k", "--num_files", type=int, default=10)
    args = parser.parse_args()

    # csv
    input_paths = args.input_paths
    output_path = args.output_path
    n = args.number_of_examples
    num_files = args.num_files

    assert num_files > 0

    output_files = [open(os.path.join(output_path, f"{i}.jsonl"), "w") for i in range(num_files)]
    ex_generator = itertools.chain.from_iterable(read_jsonl(path) for path in input_paths.split(","))

    for i, ex in tqdm(enumerate(ex_generator), total=n):
        # only keep id and text for this splitting step
        ex_jsonl = json.dumps({"id": ex["id"], "text": ex["text"]}) + "\n"
        ex_file_idx = i % num_files
        output_file = output_files[ex_file_idx]
        output_file.write(ex_jsonl)

    for file in output_files:
        file.close()


if __name__ == "__main__":
    main()
