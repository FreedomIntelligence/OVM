#!/bin/bash

generator_id=llama7b-2-ep2
model_name_or_path=~/models/game24/generators/${generator_id}

accelerate launch \
  --main_process_port=20659 \
  generate_paths_and_eval.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset game24 \
  --data_dir data/game24 \
  --output_dir eval_results/game24/generator \
  --metric_output_dir eval_results/game24/generator \
  --target_set mid \
  --batch_size 32 \
  --do_sample False \
  --max_new_tokens 400 \
  --seed 42
