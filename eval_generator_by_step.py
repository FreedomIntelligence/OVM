from utils.states import set_random_seed
from utils.models import load_model
from utils.verifier_models import load_generator_and_verifier
from utils.datasets import make_testing_dataloader
from utils.sampling import SamplingWithCalculator


import torch
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from typing import Optional, List, Dict, Set, Any, Union
from accelerate import Accelerator
import os
import json
import gc
import numpy as np



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    verifier_model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

    fp16: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    dataset: str = field(default='gsm8k')

    data_dir: str = field(default='data/gsm8k/model_generation', metadata={"help": "Path to the training data."})
    target_set: str = field(default='test', metadata={"help": "specify which data set to generate"})
    
    output_dir: str = field(default='eval_results/gsm8k/generator_with_verifier', metadata={"help": "Path to save the responses and metrics."})

@dataclass
class GenerationArguments:
    do_sample: bool = field(default=False)
    num_beams: int = field(default=1)

    temperature: float = field(default=0.7)
    top_k: int = field(default=50)
    top_p: float = field(default=1.0)
    repetition_penalty: float = field(default=1.0)
    length_penalty: float = field(default=1.0)

    max_length : int = field(default=2048)
    max_new_tokens: int = field(default=-1)

@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)
    vs_batch_size: int = field(default=1)
    n_sampling_steps: int = field(default=10)
    n_beam: int = field(default=1)
    max_step_length: int = field(default=100)
    max_n_step: int = field(default=10)

    inference_mode: str = field(default='beam')
    dedup_mode: int = field(default=0)

    seed: int = field(default=None)


def get_save_files(model_args: dataclass, data_args: dataclass, inference_args: dataclass):
    output_dir = os.path.join(data_args.output_dir, data_args.target_set)

    if inference_args.inference_mode == 'beam':
        verifier_id = os.path.basename(os.path.normpath(model_args.verifier_model_name_or_path))
        output_dir = os.path.join(output_dir, verifier_id)
        os.makedirs(output_dir, exist_ok=True)

    inference_mode = inference_args.inference_mode
    if inference_args.inference_mode == 'beam':
        inference_mode = {
            0: 'beam',
            1: 'beaml',
        }[inference_args.dedup_mode] + str(inference_args.n_beam)

    seed_mode = ''
    if inference_args.seed is not None and inference_args.seed != 42:
        seed_mode = str(inference_args.seed)

    generator_id = os.path.basename(os.path.normpath(model_args.model_name_or_path))

    suffix = f'_%step{inference_args.n_sampling_steps}_{generator_id}_{inference_mode}_{seed_mode}'.strip('_')
    responses_file = f"responses_{suffix}.jsonl"
    metrics_file = f"metrics_{suffix}.json"

    return os.path.join(output_dir, responses_file), os.path.join(output_dir, metrics_file)



def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, InferenceArguments))
    model_args, data_args, generation_args, inference_args = parser.parse_args_into_dataclasses()
    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)
    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)

    if data_args.dataset == 'gsm8k':
        from utils.gsm8k.datasets import make_test_generator_data_module
        from utils.gsm8k.decoding import extract_answer, get_answer_label
        from utils.gsm8k.metrics import GeneratorAnswerAcc

    elif data_args.dataset == 'game24':
        from utils.game24.datasets import make_test_generator_data_module
        from utils.game24.decoding import extract_expression, get_answer_label
        from utils.game24.metrics import GeneratorAnswerAcc
        extract_answer = extract_expression
    else:
        raise NotImplementedError

    responses_file, metrics_file = get_save_files(model_args, data_args, inference_args)

    accelerator = Accelerator()
    
    if inference_args.inference_mode == 'beam':
        generator, verifier, tokenizer = load_generator_and_verifier(model_args)

    else:
        raise NotImplementedError

    dataset = make_test_generator_data_module(tokenizer, data_args, inference_args)
    dataloader = make_testing_dataloader(dataset, batch_size=1)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)

    sampler = SamplingWithCalculator(accelerator=accelerator, model=generator, verifier=verifier, tokenizer=tokenizer, generation_args=generation_args)
    acc_metric = GeneratorAnswerAcc(n_data=len(dataset))

    generator.eval().cuda()
    accelerator.unwrap_model(generator).gradient_checkpointing_enable()
    if verifier is not None:
        verifier.eval().cuda()
        accelerator.unwrap_model(verifier).gradient_checkpointing_enable()
    accelerator.wait_for_everyone()


    response_list = [
        {
            'idx': data['idx'],
            'input': data['input'],
            'question': data['question'],
            **data['record_data'],
        } 
        for data in dataset
    ]

    progress = tqdm(total=len(dataloader)) if accelerator.is_main_process else None
    all_idxs_list, all_references_list, all_completions_list, all_intermediates_list  =  tuple([] for _ in range(4))
    for _, batch in enumerate(dataloader):
        idx = batch['idx'][0]
        inp = batch['input'][0]
        reference = batch['reference'][0]

        completion, intermediates = sampler.sample_by_steps(
            qn_str=inp,
            batch_size=inference_args.batch_size,
            vs_batch_size=inference_args.vs_batch_size,
            n_beam=inference_args.n_beam,
            n_sampling_steps=inference_args.n_sampling_steps,
            max_step_length=inference_args.max_step_length,
            max_n_step=inference_args.max_n_step,
            inference_mode=inference_args.inference_mode,
            dedup_mode=inference_args.dedup_mode,
        )

        acc_metric([completion], [reference])

        for obj, container in [
            (idx, all_idxs_list), 
            (reference, all_references_list), 
            (completion, all_completions_list),
            (intermediates, all_intermediates_list),
        ]:
            container.append(obj)

        if accelerator.is_main_process:
            progress.update(1)

        gc.collect(); torch.cuda.empty_cache()


    # gather
    if accelerator.num_processes != 1:
        all_idxs_gather, all_references_gather, all_completions_gather, all_intermediates_gather  =  tuple([None] * dist.get_world_size() for _ in range(4))
        for obj, container in [
            (all_idxs_list, all_idxs_gather), 
            (all_references_list, all_references_gather), 
            (all_completions_list, all_completions_gather),
            (all_intermediates_list, all_intermediates_gather),
        ]:
            dist.all_gather_object(container, obj)

        all_idxs_gather, all_references_gather, all_completions_gather, all_intermediates_gather = tuple([item for sublist in container for item in sublist]
                                                    for container in [all_idxs_gather, all_references_gather, all_completions_gather, all_intermediates_gather])
    else:
        all_idxs_gather, all_references_gather, all_completions_gather, all_intermediates_gather = all_idxs_list, all_references_list, all_completions_list, all_intermediates_list


    # record
    for idx, reference, completion, intermediates in zip(all_idxs_gather, all_references_gather, all_completions_gather, all_intermediates_gather):
        if 'response' in response_list[idx]:
            continue
        
        response_answer = extract_answer(completion)
        response_list[idx].update({
            'response': completion,
            'response_answer': response_answer,
            'label': get_answer_label(response_answer, reference),
            'intermediate_steps': intermediates,
        })


    # save outputs
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(responses_file), exist_ok=True)
        with open(responses_file, 'w') as fp:
            fp.writelines([json.dumps(data) + '\n'  for data in response_list])
        print(f"+ [Save] Save Responses to {responses_file}")


    # calculate metrics
    metrics = {
        'accuracy': acc_metric.get_metric()
    }
    accelerator.print(metrics)


    # save metrics
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        json.dump(metrics, open(metrics_file,'w'), indent=4, ensure_ascii=False)
        print(f"+ [Save] Save Metrics to {metrics_file}")



if __name__ == "__main__":
    main()

