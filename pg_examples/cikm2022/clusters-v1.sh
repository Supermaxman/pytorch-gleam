#!/usr/bin/bash

# TODO specify i/o in cmd
gleam predict \
	--config pg_examples/cikm2022/ct-v11-pred.yaml \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=/users/max/data/models/ct/ct-v11/predictions-test \
	--data.ContrastiveFrameDataModule \
	--data.label_name=candidates \
	--data.predict_path=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_candidates_1_19.json \
	--data.predict_mode=same


python pytorch_gleam/scripts/contrastive_cluster.py \
	-i /shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_candidates_1_19.json \
	-p /users/max/data/models/ct/ct-v11/predictions/predictions-Q1_Q19.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/clusters-Q1_Q19.jsonl \
	--threshold 4.0 \
	--clustering complete


python pytorch_gleam/scripts/contrastive_framing.py \
	-i /users/max/data/models/ct/ct-v11/predictions/clusters-Q1_Q19.jsonl \
	-o /users/max/data/models/ct/ct-v11/predictions/cluster-framings-Q1_Q19.jsonl

# TODO specify i/o in cmd
gleam predict --config pg_examples/cikm2022/ct-v11-pred-v3.yaml
