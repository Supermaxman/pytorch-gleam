import argparse
import os

import torch
from transformers import BertConfig, BertModel, BertTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--checkpoint_path", required=True)
    parser.add_argument("-m", "--model_name", required=True)
    parser.add_argument("-n", "--pre_model_name", required=True)
    parser.add_argument("-o", "--output_path", required=True)
    parser.add_argument("-p", "--prefix_count", default=2, type=int)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint_path
    model_name = args.model_name
    pre_model_name = args.pre_model_name
    output_path = args.output_path
    prefix_count = args.prefix_count

    print("Loading config...")
    config = BertConfig.from_pretrained(pre_model_name)
    print("Loading tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(pre_model_name)

    print("Loading model...")
    model = BertModel(config)

    print("Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    weights = {".".join(k.split(".")[prefix_count:]): v for k, v in checkpoint["state_dict"].items()}
    keys = model.load_state_dict(weights, strict=False)
    if len(keys.missing_keys) > 0:
        print("Missing keys:")
        for key in keys.missing_keys:
            print(f"  {key}")

    if len(keys.unexpected_keys) > 0:
        print("Unexpected keys:")
        for key in keys.unexpected_keys:
            print(f"  {key}")

    print("Saving model...")
    model_path = os.path.join(output_path, model_name)

    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path, save_config=True)
    print(model_path)


if __name__ == "__main__":
    main()
