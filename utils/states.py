import torch
import os
import json
from dataclasses import dataclass
import random
import math
import numpy as np
from accelerate import Accelerator


def set_deepspeed_config(accelerator: Accelerator, training_args: dataclass):
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = training_args.per_device_train_batch_size
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = training_args.per_device_train_batch_size * world_size * accelerator.gradient_accumulation_steps


def set_training_states(data_module: dict, training_args: dataclass):
    set_num_steps_per_epoch(data_module, training_args)
    set_num_training_steps(training_args)
    set_num_updating_steps(training_args)
    set_num_eval_steps(training_args)
    set_per_eval_steps(training_args)
    set_num_warmup_steps(training_args)

    set_num_logging_steps(training_args)
    set_per_save_steps(training_args)
    
    print(f"+ [Training States] There are {training_args.num_training_steps} steps in total.")


def set_num_steps_per_epoch(data_module: dict, training_args: dataclass):
    num_devices = int(os.environ.get("WORLD_SIZE", 1))

    len_train_set_per_device = math.ceil(len(data_module["train_dataset"]) / num_devices)
    num_train_steps_per_device = math.ceil(len_train_set_per_device / training_args.per_device_train_batch_size)
    num_updating_steps_per_epoch = num_train_steps_per_device // training_args.gradient_accumulation_steps

    len_eval_set_per_device = math.ceil(len(data_module["val_dataset"]) / num_devices) if data_module["val_dataset"] is not None else None
    num_eval_steps_per_device = math.ceil(len_eval_set_per_device / training_args.per_device_eval_batch_size) if data_module["val_dataset"] is not None else None

    training_args.num_training_steps_per_epoch = num_train_steps_per_device
    training_args.num_updating_steps_per_epoch = num_updating_steps_per_epoch
    training_args.num_eval_steps_per_epoch = num_eval_steps_per_device

def set_num_training_steps(training_args: dataclass):
    if training_args.max_steps != -1:
        num_training_steps = training_args.max_steps
    else:
        assert training_args.num_train_epoches != -1
        num_training_steps = training_args.num_training_steps_per_epoch * training_args.num_train_epoches
    num_training_steps_aggr_devices = num_training_steps * int(os.environ.get("WORLD_SIZE", 1))

    training_args.num_training_steps = num_training_steps
    training_args.num_training_steps_aggr_devices = num_training_steps_aggr_devices

def set_num_updating_steps(training_args: dataclass):
    num_updating_steps = training_args.num_training_steps // training_args.gradient_accumulation_steps
    num_updating_steps_aggr_devices = num_updating_steps * int(os.environ.get("WORLD_SIZE", 1))

    training_args.num_updating_steps = num_updating_steps
    training_args.num_updating_steps_aggr_devices = num_updating_steps_aggr_devices
        

def set_num_eval_steps(training_args: dataclass):
    training_args.num_eval_steps = training_args.num_eval_steps_per_epoch

def set_per_eval_steps(training_args: dataclass):
    if training_args.eval_steps != -1:
        per_eval_steps = training_args.eval_steps
    else:
        assert training_args.eval_epoches != -1
        per_eval_steps = training_args.num_training_steps_per_epoch * training_args.eval_epoches
    
    training_args.per_eval_steps = per_eval_steps

def set_num_warmup_steps(training_args: dataclass):
    # if training_args.warmup_steps != -1:
    #     num_warmup_steps_forward = training_args.warmup_steps
    # else:
    #     assert training_args.warmup_ratio != -1
    #     num_warmup_steps_forward = int(training_args.num_training_steps * training_args.warmup_ratio)
    # num_updating_warmup_steps = num_warmup_steps_forward // training_args.gradient_accumulation_steps
    # num_updating_warmup_steps_aggr_devices = num_updating_warmup_steps * int(os.environ.get("WORLD_SIZE", 1))
    if training_args.warmup_steps != -1:
        num_updating_warmup_steps = training_args.warmup_steps
    else:
        assert training_args.warmup_ratio != -1
        num_updating_warmup_steps = int(training_args.num_updating_steps * training_args.warmup_ratio)
    num_updating_warmup_steps_aggr_devices = num_updating_warmup_steps * int(os.environ.get("WORLD_SIZE", 1))

    training_args.num_updating_warmup_steps = num_updating_warmup_steps
    training_args.num_updating_warmup_steps_aggr_devices = num_updating_warmup_steps_aggr_devices

def set_num_logging_steps(training_args: dataclass):
    if training_args.logging_steps != -1:
        num_logging_steps = training_args.logging_steps
    else:
        assert training_args.logging_epoches != -1
        num_logging_steps = training_args.num_training_steps_per_epoch * training_args.logging_epoches
    
    training_args.num_logging_steps = num_logging_steps

def set_per_save_steps(training_args: dataclass):
    if training_args.save_steps != -1:
        per_save_steps = training_args.save_steps
    else:
        assert training_args.save_epoches != -1
        per_save_steps = training_args.num_training_steps_per_epoch * training_args.save_epoches
    
    training_args.per_save_steps = per_save_steps


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


