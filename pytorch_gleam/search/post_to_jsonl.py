import argparse

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


def create_jsonl_doc(post):
    post_id = post["id"]
    post_text = post["text"]
    doc = {"id": post_id, "contents": post_text}
    return doc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    parser.add_argument("-o", "--output_path", required=True)

    args = parser.parse_args()

    print("Writing jsonl posts...")
    write_jsonl(tqdm((create_jsonl_doc(tweet) for tweet in read_jsonl(args.input_path))), args.output_path)

    print("Done!")


if __name__ == "__main__":
    main()
