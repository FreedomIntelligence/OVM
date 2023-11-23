#!/bin/bash

generator_id=llama7b-2-ep2
model_name_or_path=~/models/gsm8k/generators/${generator_id}

accelerate launch \
  --main_process_port=20659 \
  generate_paths_and_eval.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset gsm8k \
  --data_dir data/gsm8k \
  --output_dir eval_results/gsm8k/generator \
  --metric_output_dir eval_results/gsm8k/generator \
  --target_set test \
  --batch_size 32 \
  --do_sample False \
  --max_new_tokens 400 \
  --seed 42
