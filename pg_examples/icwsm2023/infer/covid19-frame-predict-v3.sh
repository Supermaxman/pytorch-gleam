#!/usr/bin/env bash

# run names
FILE_ID=covid19-frame-predict-v3

data_version=v11
data_root=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter
artifacts_path=${data_root}/${data_version}/artifacts

export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.16.1.1-1.fc36.x86_64/

index_data_path=${artifacts_path}/tweets-index-data
small_index_path=${artifacts_path}/tweets-index-small
frame_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames-covid19-parsed.jsonl

output_path=${artifacts_path}/${FILE_ID}/${FILE_ID}

mkdir -p ${artifacts_path}
mkdir -p ${artifacts_path}/${FILE_ID}
mkdir -p ${index_data_path}

gleam-tweet-to-jsonl \
 --input_path ${data_root}/${data_version}/tweets.jsonl \
 --output_path ${index_data_path}/tweets.jsonl

python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input ${index_data_path} \
  -index ${small_index_path} \
  -storeDocvectors

gleam-search-tweet-index \
  --index_path ${small_index_path} \
  --query_path ${frame_path} \
  --output_path ${output_path}_bm25_scores.json \
  --top_k 1000000 \
  --threads 8

gleam-rerank \
 --index_path ${index_data_path} \
 --questions_path ${frame_path} \
 --scores_path ${output_path}_bm25_scores.json \
 --output_path ${output_path}_rerank_scores \
 --pre_model_name nboost/pt-biobert-base-msmarco \
 --batch_size 64 \
 --max_seq_len 128 \
 --gpus 0

python pytorch_gleam/search/select_candidates.py \
  --data_path ${index_data_path} \
  --scores_path ${output_path}_rerank_scores \
  --output_path ${output_path}_candidates.jsonl \
  --min_score 2.0

# ( sleep 7200 ; python pytorch_gleam/search/select_candidates.py \
#   --data_path ${index_data_path} \
#   --scores_path ${output_path}_rerank_scores \
#   --output_path ${output_path}_candidates.jsonl \
#   --min_score 1.0; python pytorch-gleam/pytorch_gleam/parse/efpparse.py \
#   --input_path ${output_path}_candidates.jsonl \
#   --frame_path ${frame_path} \
#   --output_path ${output_path}_candidates_parsed.jsonl \
#   --num_processes 8 ) &

python pytorch_gleam/parse/efpparse.py \
  --input_path ${output_path}_candidates.jsonl \
  --frame_path ${frame_path} \
  --output_path ${output_path}_candidates_parsed.jsonl \
  --num_processes 12

python pytorch_gleam/ex/gleam.py predict --config pg_examples/icwsm2023/infer/mcfmgcn-v48-predict.yaml

python pytorch_gleam/stance/frame_stance_ts.py \
  --input_path ${output_path}_stance-predictions \
  --tweet_path ${data_root}/${data_version}/tweets.jsonl \
  --output_path ${output_path}_stance-ts.jsonl \
  --threshold 0.3293
