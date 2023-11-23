#!/bin/bash

generator_id=llama7b-2-ep2
n_solutions=100

model_name_or_path=~/models/game24/generators/${generator_id}

accelerate launch \
  --main_process_port=20658 \
  generate_paths_and_eval.py \
  --model_name_or_path ${model_name_or_path} \
  --dataset game24 \
  --data_dir data/game24 \
  --output_dir data/game24/model_generation \
  --metric_output_dir eval_results/game24/generator \
  --target_set train \
  --n_solutions ${n_solutions} \
  --batch_size 16 \
  --do_sample True \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 1.0 \
  --max_new_tokens 400
