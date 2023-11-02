#!/usr/bin/env bash

# run names
FILE_ID=covid19-frame-rel-v1

artifacts_path=/shared/aifiles/disk1/covid19/artifacts/fb
data_version=v3
data_root=/users/max/data/corpora/covid19-vaccine-facebook

# export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.11.0.9-0.fc32.x86_64/
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-11.0.16.1.1-1.fc36.x86_64

index_data_path=${data_root}/${data_version}/jsonl/tweets-index-data-v1
small_index_path=${data_root}/${data_version}/jsonl/tweets-small-index-v1
frame_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/frames-covid19-parsed.jsonl

mkdir -p ${artifacts_path}
mkdir -p ${artifacts_path}/${FILE_ID}

output_path=${artifacts_path}/${FILE_ID}/${FILE_ID}
mkdir -p ${index_data_path}

python pytorch_gleam/search/post_to_jsonl.py \
  --input_path ${data_root}/${data_version}/posts.jsonl \
  --output_path ${index_data_path}/posts.jsonl


python -m pyserini.index \
  -collection JsonCollection \
  -generator DefaultLuceneDocumentGenerator \
  -threads 8 \
  -input ${index_data_path} \
  -index ${small_index_path} \
  -storeDocvectors

python pytorch_gleam/search/index_search.py \
  --index_path ${small_index_path} \
  --query_path ${frame_path} \
  --output_path ${output_path}_bm25_scores.json \
  --top_k 10000 \
  --threads 8

python pytorch_gleam/search/rerank.py \
 --index_path ${index_data_path} \
 --questions_path ${frame_path} \
 --scores_path ${output_path}_bm25_scores.json \
 --output_path ${output_path}_rerank_scores \
 --pre_model_name nboost/pt-biobert-base-msmarco \
 --batch_size 64 \
 --max_seq_len 128 \
 --gpus 0

python pytorch_gleam/search/select_candidates.py \
  --data_path ${data_root}/${data_version}/posts.jsonl \
  --scores_path ${output_path}_rerank_scores \
  --output_path ${output_path}_candidates.jsonl \
  --min_score 2.0 \
  --count 8735966

echo ${output_path}_candidates.jsonl
