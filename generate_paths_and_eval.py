from utils.states import set_random_seed
from utils.models import load_model
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



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    fp16: Optional[bool] = field(default=False)

@dataclass
class DataArguments:
    dataset: str = field(default='gsm8k')

    data_dir: str = field(default='data/game24/', metadata={"help": "Path to the training data."})
    target_set: str = field(default='hard', metadata={"help": "specify which data set to generate"})
    
    output_dir: str = field(default='data/game24/model_generation', metadata={"help": "Path to save the responses and metrics."})
    metric_output_dir: str = field(default='eval_results/game24', metadata={"help": "Path to save the responses and metrics."})

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
    n_solutions: int = field(default=1)
    
    seed: int = field(default=None)


def get_save_files(model_args: dataclass, data_args: dataclass, inference_args: dataclass):
    output_dir = os.path.join(data_args.output_dir, data_args.target_set)
    metric_output_dir = os.path.join(data_args.metric_output_dir, data_args.target_set)
    
    model_id = os.path.basename(os.path.normpath(model_args.model_name_or_path))
    n_solution_suffix = f"n{inference_args.n_solutions}" if inference_args.n_solutions > 1 else ""

    suffix = f'_{n_solution_suffix}_{model_id}'.strip('_')
    responses_file = f"responses_{suffix}.jsonl"
    metrics_file = f"metrics_{suffix}.json"
    return os.path.join(output_dir, responses_file), os.path.join(metric_output_dir, metrics_file)




def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, InferenceArguments))
    model_args, data_args, generation_args, inference_args = parser.parse_args_into_dataclasses()
    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)
        
    if data_args.dataset == 'gsm8k':
        from utils.gsm8k.datasets import make_test_generator_data_module
        from utils.gsm8k.decoding import extract_answer, get_answer_label
        from utils.gsm8k.metrics import GeneratorAnswerAcc, MultiSamplingAnswerAcc
    elif data_args.dataset == 'game24':
        from utils.game24.datasets import make_test_generator_data_module
        from utils.game24.decoding import extract_expression, get_answer_label
        from utils.game24.metrics import GeneratorAnswerAcc, MultiSamplingAnswerAcc
        extract_answer = extract_expression
    else:
        raise NotImplementedError


    accelerator = Accelerator()
    
    model, tokenizer = load_model(model_args)
    dataset = make_test_generator_data_module(tokenizer, data_args, inference_args)
    dataloader = make_testing_dataloader(dataset, batch_size=inference_args.batch_size)
    
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)

    sampler = SamplingWithCalculator(accelerator=accelerator, model=model, tokenizer=tokenizer, generation_args=generation_args)
    greedy_metric = GeneratorAnswerAcc(n_data=len(dataset))
    sampling_metric = MultiSamplingAnswerAcc(n_data=len(dataset))


    response_list = [
        {
            'idx': data['idx'],
            'input': data['input'],
            'question': data['question'],
            **data['record_data'],
            'outputs': [],
        } 
        for data in dataset
    ]
    responses_file, metrics_file = get_save_files(model_args, data_args, inference_args)
    assert not os.path.exists(responses_file), f'{responses_file} has existed!'

    model.eval().cuda()
    accelerator.unwrap_model(model).gradient_checkpointing_enable()
    for i in range(inference_args.n_solutions):
        accelerator.wait_for_everyone()
        progress = tqdm(total=len(dataloader), desc='{}-th/{} solution'.format(i+1, inference_args.n_solutions)) if accelerator.is_main_process else None

        all_idxs_list, all_references_list, all_completions_list  =  tuple([] for _ in range(3))
        sampling_metric.start_new_sol_epoch()
        for _, batch in enumerate(dataloader):
            idx_list, input_list, reference_list = tuple(batch[k] for k in ('idx', 'input', 'reference'))

            completions = sampler.sample(input_list)
            
            greedy_metric(completions, reference_list)
            sampling_metric(completions, reference_list)

            for obj, container in [
                (idx_list, all_idxs_list), 
                (reference_list, all_references_list),
                (completions, all_completions_list),
            ]:
                container.extend(obj)

            if accelerator.is_main_process:
                progress.update(1)

        sampling_metric.end_the_sol_epoch()

        gc.collect(); torch.cuda.empty_cache()


        # gather
        if accelerator.num_processes != 1:
            all_idxs_gather, all_references_gather, all_completions_gather  =  tuple([None] * dist.get_world_size() for _ in range(3))
            for obj, container in [
                (all_idxs_list, all_idxs_gather), 
                (all_references_list, all_references_gather), 
                (all_completions_list, all_completions_gather)
            ]:
                dist.all_gather_object(container, obj)

            all_idxs_gather, all_references_gather, all_completions_gather = tuple([item for sublist in container for item in sublist]
                                                        for container in [all_idxs_gather, all_references_gather, all_completions_gather])
        else:
            all_idxs_gather, all_references_gather, all_completions_gather = all_idxs_list, all_references_list, all_completions_list


        # record
        for idx, reference, completion in zip(all_idxs_gather, all_references_gather, all_completions_gather):
            if len(response_list[idx]['outputs']) == i+1:
                continue

            response_answer = extract_answer(completion)
            response_list[idx]['outputs'].append({
                'response': completion,
                'response_answer': response_answer,
                'label': get_answer_label(response_answer, reference),
            })


        # save outputs
        if accelerator.is_main_process:
            os.makedirs(os.path.dirname(responses_file), exist_ok=True)
            with open(responses_file, 'w') as fp:
                fp.writelines([json.dumps(data) + '\n'  for data in response_list])
            print(f"+ [Save] Save Responses to {responses_file}")


    # save greedy outputs
    if accelerator.is_main_process and inference_args.n_solutions == 1:
        for data in response_list:
            outputs = data.pop('outputs')
            data = data.update(**outputs[0])

        with open(responses_file, 'w') as fp:
            fp.writelines([json.dumps(data) + '\n'  for data in response_list])
        print(f"+ [Save] Save Responses to {responses_file}")
            


    # calculate metrics
    if inference_args.n_solutions == 1:
        metrics = {
            'accuracy': greedy_metric.get_metric()
        }
    else:
        pass_k, acc_majority = sampling_metric.get_metric(inference_args.n_solutions)
        metrics = {
            'multiple guesses': pass_k,
            'self consistency': acc_majority,
        }
    accelerator.print(metrics)


    # save metrics
    if accelerator.is_main_process:
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        json.dump(metrics, open(metrics_file,'w'), indent=4, ensure_ascii=False)
        print(f"+ [Save] Save Metrics to {metrics_file}")



if __name__ == "__main__":
    main()

