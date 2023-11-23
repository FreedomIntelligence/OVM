#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=GAME24-Generator-Finetune
export WANDB_ENTITY=


model_name_or_path=meta-llama/Llama-2-7b-hf
save_generator_id=llama7b-2-ep2

save_dir=~/models/game24/generators/${save_generator_id}/
export WANDB_NAME=${save_generator_id}




accelerate launch \
  --config_file ./configs/zero1.yaml \
  --main_process_port=20650 \
  train_generator.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset game24 \
  --data_dir data/game24 \
  --target_set train \
  --save_dir ${save_dir} \
  --num_train_epoches 2 \
  --eval_steps 200 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_steps 200 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ./wandb \
  --logging_steps 8 \
  --seed 42
  
