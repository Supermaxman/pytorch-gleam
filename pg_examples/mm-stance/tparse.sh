#!/usr/bin/env bash

# python pytorch_gleam/parse/fparse.py \
#   --input_path /shared/aifiles/disk1/media/twitter/v10/data/frames.json \
#   --output_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json


python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12


python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-tparsed.jsonl \
  --label_name labels \
  --num_processes 12


python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/train-caption-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/dev-caption-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12

python pytorch_gleam/parse/efpparse.py \
  --input_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr.jsonl \
  --frame_path /shared/aifiles/disk1/media/twitter/v10/data/frames-parsed.json \
  --output_path /shared/aifiles/disk1/media/twitter/v10/data/test-caption-ocr-tparsed.jsonl \
  --label_name labels \
  --num_processes 12
