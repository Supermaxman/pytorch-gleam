#!/usr/bin/bash


input_path=/shared/hltdir4/disk1/team/data/corpora/covid19-vaccine-twitter/v4/jsonl-non-rt/covid_candidates_1_40.json
frame_path=/shared/hltdir4/disk1/team/data/corpora/co-vax-frames/covid19/co-vax-frames.json

prediction_name=Q1_Q40-v1
model_name=ct-v11
save_path=/users/max/data/models/ct
model_path=${save_path}/${model_name}
cluster_pred_path=${model_path}/predictions-same-${prediction_name}
frame_pred_path=${model_path}/predictions-other-${prediction_name}
frame_compare_path=${model_path}/predictions-frames-${prediction_name}


gleam predict \
	--config pg_examples/cikm2022/${model_name}-pred.yaml \
	--trainer.callbacks=TPURichProgressBar \
	--trainer.callbacks=FitCheckpointCallback \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=${cluster_pred_path} \
	--data=ContrastiveFrameDataModule \
	--data.init_args.label_name=candidates \
	--data.init_args.predict_path=${input_path} \
	--data.init_args.predict_mode=same


python pytorch_gleam/scripts/contrastive_cluster.py \
	-i ${input_path} \
	-p ${cluster_pred_path}/predictions.jsonl \
	-o ${cluster_pred_path}/clusters.jsonl \
	--threshold 12.0 \
	--min_cluster_size 2 \
	--clustering complete \
;python pytorch_gleam/scripts/contrastive_framing.py \
	-i ${cluster_pred_path}/clusters.jsonl \
	-o ${cluster_pred_path}/cluster-framings.jsonl

gleam predict \
	--config pg_examples/cikm2022/ct-v11-pred.yaml \
	--trainer.callbacks=TPURichProgressBar \
	--trainer.callbacks=FitCheckpointCallback \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=${frame_pred_path} \
	--data.init_args.label_name=questions \
	--data.init_args.predict_path=${cluster_pred_path}/cluster-framings.jsonl \
	--data.init_args.predict_mode=other

python pytorch_gleam/scripts/contrastive_question_cluster.py \
	-i ${cluster_pred_path}/cluster-framings.jsonl \
	-p ${frame_pred_path}/predictions.jsonl \
	-o ${frame_pred_path}/question-clusters.jsonl \
	--threshold 12.0 \
	--clustering complete \
;python pytorch_gleam/scripts/contrastive_question_framing.py \
	-i ${frame_pred_path}/question-clusters.jsonl \
	-o ${frame_pred_path}/question-cluster-framings.jsonl



gleam predict \
	--config pg_examples/cikm2022/ct-v11-pred.yaml \
	--trainer.callbacks=TPURichProgressBar \
	--trainer.callbacks=FitCheckpointCallback \
	--trainer.callbacks=JsonlWriter \
	--trainer.callbacks.output_path=${frame_compare_path} \
	--data.init_args.label_name=questions \
	--data.init_args.frame_path=${frame_path} \
	--data.init_args.predict_path=${frame_pred_path}/question-cluster-framings.jsonl \
	--data.init_args.predict_mode=frames


python pytorch_gleam/scripts/contrastive_compare_framing.py \
	-i ${frame_pred_path}/question-cluster-framings.jsonl \
	-f ${frame_path} \
	-p ${frame_compare_path}/predictions.jsonl \
	-o ${frame_compare_path}/question-cluster-framings-compare.jsonl

python pytorch_gleam/scripts/contrastive_compare_framing_manual.py \
  -i ${frame_compare_path}/question-cluster-framings-compare.jsonl \
  -o ${frame_compare_path}/${model_name}-${prediction_name}.xlsx


python pytorch_gleam/scripts/contrastive_compare_framing_stats.py \
	-f ${frame_path} \
  -i ${frame_compare_path}/${model_name}-${prediction_name}.xlsx
