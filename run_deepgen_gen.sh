#!/bin/bash
set -e
for EXP_NAME in rl; do
  CUDA_VISIBLE_DEVICES=0 python main.py \
  --train_dir '' \
  --scene pegasus_xsum_generation \
  --eval_dir static_data/xsum/xsum_test.tsv \
  --previous_dir . \
  --output_dir ./pegasus_generation_${EXP_NAME}/  \
  --do_predict --do_eval \
  --per_device_eval_batch_size 20 \
  --model google/pegasus-xsum \
  --trainer common \
  --num_beam_groups 1 \
  --diversity_penalty 0 \
  --recover xsum_${EXP_NAME}_model \
  --num_return_sequences 1 \
  --outputter_num_workers 0 \
  --early_stopping True \
  --no_repeat_ngram_size 3
done