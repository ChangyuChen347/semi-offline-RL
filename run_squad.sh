#!/bin/bash
CUDA_VISIBLE_DEVICES=2 python main.py \
    --do_train \
    --scene t5_squad_generation \
    --use_logit True \
    --report_to tensorboard \
    --seed 2022 \
    --smooth 0.15 \
    --trainer rl \
    --save_steps 10000000 \
    --batch_mean False \
    --learning_rate 0.000003 \
    --num_train_epochs 60 \
    --max_grad_norm 1 \
    --print_every 1000 \
    --save_every 4000 \
    --eval_steps 1000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --per_device_eval_batch_size 32 \
    --length_normalize_4_rl True \
    --training_length_penalty 1 \
    --train_dir static_data/squad/squad_train.tsv \
    --eval_dir static_data/squad/squad_test.tsv \
    --cand_pos_remove_sp_tk True \
    --recover squad_base_model \
    --exp_name demo_squad \
    --rewards squad_rouge_bleu \
    --rl_weight 4 \
    --sample_num 15 \
    --mask_rate 0.4 \
    --kd_inputs_worst True \
    --eval_metrics squad_bleu,squad_rouge \
    --early_stopping False \
    --seq_decode_model t5 \
