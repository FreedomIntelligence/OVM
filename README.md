# OVM, Outcome-supervised Value Models for Planning in Mathematical Reasoning


Code, metrics, and models for the paper [Outcome-supervised Verifiers for Planning in Mathematical Reasoning](https://arxiv.org/pdf/2311.09724.pdf)


The key technical implementations (`utils/sampling.py`):

1. **Value-guided beam search**: step-level beam search guided by a value model

2. **Allow batch generation with caculator using cache** (2-3 times faster than a naive implementation)


## Something ...


1. Directories
- `configs`: for model training with `accelerate`
- `data`: benchmark, and generator created data for training the value model
- `eval_results`: metrics and responses
    - `generator`: generator-only (greedy, self-consistency, or pass@k)
    - `verifier`: ORM accuracy
    - `generator_with_verifier`: guided beam search, i.e. OVM and PRM
- `scripts`: scripts for training and inference
- `utils`: functions and classes


2. target_set
- GSM8K: there are `train` and `test`, which corresponds to training set and test set respectively
- Game of 24: there are `train` and `mid`
    - `train`: the first 900 problems
    - `mid`: problems index 901-1000

3. scripts for GSM8K and Game of 24 are similar. For simplicity, we only take GSM8K as the example below. You can simply run the same pipeline in Game of 24 by replacing `gsm8k` with `game24`



## Training

### Train the generator

Training data for generator:
- GSM8K: `data/gsm8k/train.jsonl`, from [OpenAI GSM8K](https://github.com/openai/grade-school-math/blob/master/grade_school_math/data/train.jsonl)
- Game of 24: `data/game24/train.jsonl`, the first 900 problems in `data/game24/24.csv` (from [ToT](https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/data/24/24.csv)) with enumerated solutions

To run the script `train_generator.sh` (under `scripts/gsm8k` or `scripts/game24`), you should first set `WANDB_API_KEY`, `WANDB_ENTITY`, `model_name_or_path`, `save_dir`. The generator is named by `save_generator_id`

```bash
cd OVM
bash scripts/gsm8k/train_generator.sh
```


### Train the OVM

#### Generation

First use the generator `generator_id` to generate `n_solutions` for each question in the training set,
```bash
cd OVM
bash scripts/gsm8k/generate.sh
```
You should first config the path of your generator checkpoint `model_name_or_path`, and set `--target_set train`

The output will be saved to `data/gsm8k/model_generation/`


#### Training

Train OVM using `train_verifier.sh`. First set `WANDB_API_KEY`, `WANDB_ENTITY`, `save_dir`, and `checkpoint_dir` (the path of generator checkpoint). The verifier is named with `save_verifier_id`
```bash
cd OVM
bash scripts/gsm8k/train_verifier.sh
```



## Inference

### Value-Guided Beam Search

Config your generator checkpoint path `model_name_or_path` and verifier checkpoint path `verifier_model_name_or_path` in `eval_step_beam.sh`
```bash
cd OVM
bash scripts/gsm8k/eval_step_beam.sh
```

(when `dedup_mode=1`, it will prioritize linguistically different candidates, which means when the sorted candidates are ['a', 'a', 'b', 'b', 'c'] it will select ['a', 'b', 'c'] rather than ['a', 'a', 'b'] if n_beam=3)

The output will be saved to `eval_results/gsm8k/generator_with_verifier/test` 
(or `eval_results/game24/generator_with_verifier/mid`)


### Vanilla Sampling with ORM

1. First sample the data: config the generator checkpoint `model_name_or_path`, and set `--target_set test`
    ```bash
    cd OVM
    bash scripts/gsm8k/generate.sh
    ```

2. Then call ORM to score and rerank the samples: config the verifier checkpoint `verifier_model_name_or_path`
    ```bash
    cd OVM
    bash scripts/gsm8k/eval_with_verifier.sh
    ```

The output will be saved to `eval_results/gsm8k/generator_with_verifier/test` 



### Greedy

Config your generator checkpoint path `model_name_or_path`
```bash
cd OVM
bash scripts/gsm8k/greedy_eval.sh
```
The output will be saved to `eval_results/gsm8k/generator/test` 

