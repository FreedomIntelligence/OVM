#!/bin/bash

generator_id=llama7b-2-ep2
verifier_id=n100-scahead-mse-lm-token-bsz128-ep10

n_beam=4
n_sampling_steps=20


model_name_or_path=~/models/game24/generators/${generator_id}
verifier_model_name_or_path=~/models/game24/verifiers/${generator_id}-${verifier_id}


accelerate launch \
  --main_process_port=20659 \
  eval_generator_by_step.py \
  --model_name_or_path ${model_name_or_path} \
  --verifier_model_name_or_path ${verifier_model_name_or_path} \
  --dataset game24 \
  --data_dir data/game24 \
  --output_dir eval_results/game24/generator_with_verifier \
  --target_set mid \
  --inference_mode beam \
  --batch_size 30 \
  --vs_batch_size 64 \
  --n_beam ${n_beam} \
  --n_sampling_steps ${n_sampling_steps} \
  --max_n_step 10 \
  --max_step_length 100 \
  --dedup_mode 1 \
  --do_sample True \
  --temperature 0.7 \
  --top_k 50 \
  --top_p 1.0 \
  --max_new_tokens 400 \
  --seed 42


