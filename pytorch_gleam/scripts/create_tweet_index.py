import argparse

from pytorch_gleam.data.jsonl import JsonlIndex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_path", required=True)
    args = parser.parse_args()

    input_path = args.input_path

    JsonlIndex.create(input_path)


if __name__ == "__main__":
    main()
