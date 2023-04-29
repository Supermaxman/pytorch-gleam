#!/usr/bin/env bash

# python pytorch_gleam/parse/fparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/frames.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json


# python pytorch_gleam/parse/tparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/train.jsonl \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-parsed.jsonl

# python pytorch_gleam/parse/tparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev.jsonl \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-parsed.jsonl

# python pytorch_gleam/parse/tparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/test.jsonl \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr-parsed.jsonl


python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-parsed.jsonl


python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr-parsed.jsonl

python pytorch_gleam/parse/tparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr.jsonl \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr-parsed.jsonl
