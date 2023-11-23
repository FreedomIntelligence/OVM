#!/bin/bash

generator_id=llama7b-2-ep2
verifier_id=n100-scahead-mse-lm-token-bsz128-ep10
verifier_model_name_or_path=~/models/game24/verifiers/${generator_id}-${verifier_id}

accelerate launch \
  --main_process_port=29510 \
  eval_with_verifier.py \
  --model_name_or_path ${verifier_model_name_or_path} \
  --data_dir data/game24/model_generation \
  --verifier_output_dir eval_results/game24/verifier \
  --generator_metric_dir eval_results/game24/generator_with_verifier \
  --generator_id ${generator_id} \
  --target_set mid \
  --batch_size 64 \
  --seed 42


