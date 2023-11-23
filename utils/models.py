from utils.flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
from utils.cached_models import build_transformers_mapping_to_cached_models, build_transformers_mapping_to_custom_tokenizers
replace_llama_attn_with_flash_attn()
build_transformers_mapping_to_cached_models()
build_transformers_mapping_to_custom_tokenizers()

from typing import Optional, List, Dict, Set, Any, Union, Callable, Mapping
from torch import nn
import torch
import pathlib
from dataclasses import dataclass
from accelerate import Accelerator
import os
import re
import shutil
from functools import wraps
import transformers
from utils.constants import DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, DEFAULT_UNK_TOKEN


def smart_tokenizer_and_embedding_resize_for_pad(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))
    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings[-1] = torch.zeros_like(input_embeddings)[0]
        output_embeddings[-1] = torch.zeros_like(output_embeddings)[0]


def build_model(model_args: dataclass, training_args: dataclass):
    # Step 1: Initialize LLM
    print(f"+ [Model] Initializing LM: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Step 2: Initialize tokenizer
    print(f"+ [Model] Initializing Tokenizer: {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    # Step 3: Add special tokens
    if tokenizer.pad_token is None:
        smart_tokenizer_and_embedding_resize_for_pad(
            special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
            tokenizer=tokenizer,
            model=model,
        )
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    # Step 4: Align special token ids between tokenizer and model.config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

def load_model(model_args):
    # Step 1: Initialize LLM
    print(f"+ [Model] Initializing LM: {model_args.model_name_or_path}")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    # Step 2: Initialize tokenizer
    print(f"+ [Model] Initializing Tokenizer: {model_args.model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="left",
        use_fast=False,
    )

    # Step 3: Add special tokens
    tokenizer.add_special_tokens({
        "eos_token": DEFAULT_EOS_TOKEN,
        "bos_token": DEFAULT_BOS_TOKEN,
        "unk_token": DEFAULT_UNK_TOKEN,
    })

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token=DEFAULT_UNK_TOKEN))

    # Step 4: Align special token ids between tokenizer and model.config
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    return model, tokenizer









def safe_delete_with_accelerator(accelerator: Accelerator, path: str):
    @accelerator.on_main_process
    def delete(path):
        shutil.rmtree(path, ignore_errors=True)
    
    delete(path)

def safe_move_with_accelerator(accelerator: Accelerator, ori_path: str, new_path: str):
    @accelerator.on_main_process
    def move(ori_path, new_path):
        try:
            shutil.move(ori_path, new_path)
        except:
            ...
        
    move(ori_path, new_path)



def wrapper_safe_save_model_with_accelerator(save_model_func):
    @wraps(save_model_func)
    def wrapper(accelerator: Accelerator,
                model: nn.Module,
                tokenizer: transformers.AutoTokenizer,
                output_dir: str):
        @accelerator.on_main_process
        def save_model(cpu_state_dict, output_dir):
            save_model_func(accelerator=accelerator, model=model, cpu_state_dict=cpu_state_dict, output_dir=output_dir)
        @accelerator.on_main_process
        def save_tokenizer(output_dir):
            tokenizer.save_pretrained(output_dir)

        os.makedirs(output_dir, exist_ok=True)
        state_dict = accelerator.get_state_dict(model)
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        
        save_model(cpu_state_dict, output_dir)
        save_tokenizer(output_dir)

        print(f"+ [Save] Save model and tokenizer to: {output_dir}")
    return wrapper


# refer to transformers.trainer._sorted_checkpoints "https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2848"
def wrapper_save_checkpoint(save_func):
    @wraps(save_func)
    def outwrapper(func):
        def wrapper(accelerator: Accelerator,
                    model: transformers.AutoModelForCausalLM,
                    tokenizer: transformers.PreTrainedTokenizer,
                    output_dir: str,
                    global_step: int,
                    save_total_limit: int=None):
            checkpoint_output_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
            if os.path.exists(checkpoint_output_dir) or save_total_limit < 1:
                return
            save_func(accelerator=accelerator, model=model, tokenizer=tokenizer, output_dir=checkpoint_output_dir)

            ordering_and_checkpoint_path = []
            glob_checkpoints = [str(x) for x in pathlib.Path(output_dir).glob('*checkpoint-*')]
            for path in glob_checkpoints:
                regex_match = re.match(r".*checkpoint-([0-9]+)", path)
                if regex_match is not None:
                    ordering_and_checkpoint_path.append((int(regex_match.group(1)), path))

            checkpoints_sorted = sorted(ordering_and_checkpoint_path)
            checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]

            best_checkpoint = [str(x) for x in pathlib.Path(output_dir).glob('best-checkpoint-*')]
            if best_checkpoint:
                best_checkpoint = best_checkpoint[0]
                best_model_index = checkpoints_sorted.index(best_checkpoint)
                for i in range(best_model_index, len(checkpoints_sorted) - 2):
                    checkpoints_sorted[i], checkpoints_sorted[i + 1] = checkpoints_sorted[i + 1], checkpoints_sorted[i]

            if save_total_limit:
                checkpoints_to_be_deleted = checkpoints_sorted[:-save_total_limit]
                for checkpoint in checkpoints_to_be_deleted:
                    safe_delete_with_accelerator(accelerator, checkpoint)
        return wrapper
    return outwrapper


def wrapper_save_best_checkpoint(save_checkpoint_func):
    @wraps(save_checkpoint_func)
    def outwrapper(func):
        def wrapper(accelerator: Accelerator,
                    model: transformers.AutoModelForCausalLM,
                    tokenizer: transformers.PreTrainedTokenizer,
                    output_dir: str,
                    global_step: int,
                    save_total_limit: int=None):  

            ori_best_checkpoint = [str(x) for x in pathlib.Path(output_dir).glob('best-checkpoint-*')]
            if ori_best_checkpoint:
                ori_best_checkpoint = ori_best_checkpoint[0]
                filename = os.path.basename(os.path.normpath(ori_best_checkpoint))[5:]
                safe_move_with_accelerator(accelerator, ori_best_checkpoint, os.path.join(output_dir, filename))

            save_checkpoint_func(accelerator=accelerator, model=model, tokenizer=tokenizer, output_dir=output_dir, global_step=global_step, save_total_limit=save_total_limit)
            checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
            best_checkpoint_dir = os.path.join(output_dir, f'best-checkpoint-{global_step}')
            safe_move_with_accelerator(accelerator, checkpoint_dir, best_checkpoint_dir)
        return wrapper
    return outwrapper





@wrapper_safe_save_model_with_accelerator
def save_llm(accelerator: Accelerator,
             model: transformers.AutoModelForCausalLM,
             cpu_state_dict: Mapping,
             output_dir: str):
    accelerator.unwrap_model(model).save_pretrained(
        output_dir,
        state_dict=cpu_state_dict,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )


@wrapper_save_checkpoint(save_func=save_llm)
def save_llm_checkpoint(accelerator: Accelerator,
                        model: transformers.AutoModelForCausalLM,
                        tokenizer: transformers.PreTrainedTokenizer,
                        checkpoint_output_dir: str):    
    ...


@wrapper_save_best_checkpoint(save_checkpoint_func=save_llm_checkpoint)
def save_best_llm_checkpoint(accelerator: Accelerator,
                             model: transformers.AutoModelForCausalLM,
                             tokenizer: transformers.PreTrainedTokenizer,
                             output_dir: str,
                             global_step: int,
                             save_total_limit: int=None):    
    ...
        


def save_training_args_with_accelerator(accelerator: Accelerator,
                                        training_args: dataclass,
                                        output_dir: str):
    output_file = os.path.join(output_dir, 'training_args.bin')
    accelerator.save(training_args, output_file)
    
    print(f"+ [Save] Save training_args to: {output_file}")




