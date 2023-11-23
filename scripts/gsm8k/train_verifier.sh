#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=GSM8K-Verifier
export WANDB_ENTITY=


n_solution=100
generator_id=llama7b-2-ep2
save_verifier_id=n${n_solution}-scahead-mse-lm-token


checkpoint_dir=~/models/gsm8k/generators/${generator_id}

final_id=${generator_id}-${save_verifier_id}
save_dir=~/models/gsm8k/verifiers/${generator_id}-${experimentID}
export WANDB_NAME=${generator_id}-${experimentID}



accelerate launch \
  --config_file ./configs/zero1.yaml \
  --main_process_port=20104 \
  train_verifier.py \
  --model_name_or_path ${checkpoint_dir} \
  --data_dir data/gsm8k/model_generation \
  --target_set train \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --dedup True \
  --per_problem_sampling_solution ${n_solution} \
  --loss_level token \
  --loss_on_llm True \
  --num_train_epoches 1 \
  --eval_steps 1000 \
  --per_device_train_batch_size 64 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_epoches 1 \
  --save_best False \
  --save_total_limit 0 \
  --logging_dir ./wandb \
  --logging_steps 20 \
  --seed 42

