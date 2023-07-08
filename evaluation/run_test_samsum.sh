#!/bin/bash
set -e
exp_name=$1
output_dir_path=$2
model_dir_path=$3
output_dir=${output_dir_path}/samsum/${exp_name}
mkdir -p ${output_dir}
CUDA_VISIBLE_DEVICES=0 python  main.py \
--train_dir static_data/samsum/samsum_train.tsv \
--scene bart_samsum_generation  \
--eval_dir static_data/samsum/samsum_test.tsv \
--previous_dir . \
--output_dir ${output_dir}/  \
--do_predict --do_eval \
--per_device_eval_batch_size 32 \
--trainer rl \
--num_return_sequences 1 \
--report_to tensorboard \
--recover ${model_dir_path}/samsum_${exp_name}_model \
