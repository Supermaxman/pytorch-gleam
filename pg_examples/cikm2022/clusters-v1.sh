#!/usr/bin/bash

# TODO take cmd input
#gleam predict --config pg_examples/cikm2022/ct-v11-pred-v1.yaml
gleam predict \
	--config pg_examples/cikm2022/ct-v11-pred.yaml \
	--trainer.callbacks=TPURichProgressBar \
	--trainer.callbacks=FitCheckpointCallback \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=/users/max/data/models/ct/ct-v11/predictions-same-Q1_Q19 \
	--data=ContrastiveFrameDataModule \
	--data.init_args.batch_size=8 \
	--data.init_args.max_seq_len=128 \
	--data.init_args.tokenizer_name=digitalepidemiologylab/covid-twitter-bert-v2 \
	--data.init_args.label_name=candidates \
	--data.init_args.predict_path=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_candidates_1_19.json \
	--data.init_args.predict_mode=same


python pytorch_gleam/scripts/contrastive_cluster.py \
	-i /shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_candidates_1_19.json \
	-p /users/max/data/models/ct/ct-v11/predictions/predictions-Q1_Q19.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/clusters-Q1_Q19.jsonl \
	--threshold 4.0 \
	--clustering complete


python pytorch_gleam/scripts/contrastive_framing.py \
	-i /users/max/data/models/ct/ct-v11/predictions/clusters-Q1_Q19.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/cluster-framings-Q1_Q19.jsonl

# TODO take cmd input
#gleam predict --config pg_examples/cikm2022/ct-v11-pred-v4.yaml
gleam predict \
	--config pg_examples/cikm2022/ct-v11-pred.yaml \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=/users/max/data/models/ct/ct-v11/predictions-all-Q1_Q19 \
	--data.init_args.label_name=questions \
	--data.init_args.predict_path=/users/max/data/models/ct/ct-v11/predictions/cluster-framings-Q1_Q19.jsonl \
	--data.init_args.predict_mode=all

python pytorch_gleam/scripts/contrastive_question_cluster.py \
	-i /users/max/data/models/ct/ct-v11/predictions/cluster-framings-Q1_Q19.jsonl \
	-p /users/max/data/models/ct/ct-v11/predictions-framings-all/predictions.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/question-clusters-Q1_Q19.jsonl \
	--threshold 4.0 \
	--clustering complete

python pytorch_gleam/scripts/contrastive_question_framing.py \
	-i /users/max/data/models/ct/ct-v11/predictions/question-clusters-Q1_Q19.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/question-cluster-framings-Q1_Q19.jsonl
