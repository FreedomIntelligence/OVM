#!/bin/bash

export WANDB_API_KEY=
export WANDB_PROJECT=GAME24-Verifier
export WANDB_ENTITY=


n_solution=100
generator_id=llama7b-2-ep2
save_verifier_id=n${n_solution}-scahead-mse-lm-token-bsz128-ep10


checkpoint_dir=~/models/game24/generators/${generator_id}

final_id=prm-${generator_id}-${save_verifier_id}
save_dir=~/models/game24/verifiers/${final_id}
export WANDB_NAME=${final_id}



accelerate launch \
  --config_file ./configs/zero1.yaml \
  --main_process_port=20104 \
  train_verifier.py \
  --model_name_or_path ${checkpoint_dir} \
  --data_dir data/game24/model_generation \
  --target_set train \
  --save_dir ${save_dir} \
  --generator_id ${generator_id} \
  --dedup True \
  --per_problem_sampling_solution ${n_solution} \
  --loss_level token \
  --loss_on_llm True \
  --process True \
  --num_train_epoches 10 \
  --eval_steps 1000 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 64 \
  --gradient_accumulation_steps 1 \
  --gradient_checkpointing True \
  --learning_rate 1e-5 \
  --weight_decay 0 \
  --lr_scheduler_type "linear" \
  --warmup_steps 0 \
  --save_epoches 2 \
  --save_best False \
  --save_total_limit 10 \
  --logging_dir ./wandb \
  --logging_steps 20 \
  --seed 42
  
