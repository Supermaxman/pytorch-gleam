#!/usr/bin/env bash

# python pytorch_gleam/parse/fparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/frames.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json


# python pytorch_gleam/parse/efpparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/train.jsonl \
#   --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-parsed.jsonl \
#   --label_name labels \
#   --num_processes 12

# python pytorch_gleam/parse/efpparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev.jsonl \
#   --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-parsed.jsonl \
#   --label_name labels \
#   --num_processes 12

# python pytorch_gleam/parse/efpparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/test.jsonl \
#   --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-parsed.jsonl \
#   --label_name labels \
#   --num_processes 12

# python pytorch_gleam/parse/efpparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr.jsonl \
#   --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr-parsed.jsonl \
#   --label_name labels \
#   --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr-parsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr-parsed.jsonl \
  --label_name labels \
  --num_processes 12


python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-parsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-parsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-parsed.jsonl \
  --label_name labels \
  --num_processes 12


python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr-parsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr-parsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr-parsed.jsonl \
  --label_name labels \
  --num_processes 12
