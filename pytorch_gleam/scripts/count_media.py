import argparse

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
    parser.add_argument("-i", "--input_paths", nargs="+", required=True)
    parser.add_argument("-t", "--total", type=int, required=True)
    args = parser.parse_args()

    input_paths = args.input_paths

    total = 0
    photos = 0
    with tqdm(total=args.total) as pbar:
        for path in input_paths:
            for ex in read_jsonl(path):
                total += 1
                if "media" in ex:
                    media = ex["media"]
                    if isinstance(media, dict):
                        media = media.values()
                    for m in media:
                        if m["type"] == "photo":
                            photos += 1
                pbar.update(1)
                photos_ratio = photos / total
                pbar.set_postfix({"photos": f"{photos}/{total} ({100* photos_ratio:.0f}%)"})

    print(f"photos: {photos}/{total} ({100* photos_ratio:.0f}%)")


if __name__ == "__main__":
    main()
