import torch
import transformers
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any, Union
import gc
from accelerate import Accelerator
import wandb
import os
import re

from utils.states import set_deepspeed_config, set_training_states, set_random_seed
from utils.optim import get_optimizers
from utils.models import build_model, save_llm_checkpoint, save_llm, save_training_args_with_accelerator
from utils.datasets import make_training_dataloaders


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    dataset: str = field(default='gsm8k')
    data_dir: str = field(default='data/gsm8k/', metadata={"help": "Path to the training data."})
    target_set: str = field(default='train')
    loss_on_prefix: bool = field(default=True, metadata={"help": "Whether to compute loss on the prefix"})

@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    max_steps: int = field(default=-1, metadata={"help": "When it is specified, num_train_epoches is ignored"})
    num_train_epoches: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)

    eval_steps: int = field(default=-1, metadata={"help": "When it is specified, eval_epoches is ignored"})
    eval_epoches: int = field(default=1)
    per_device_eval_batch_size: int = field(default=4)

    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0)
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(default=-1, metadata={"help": "When it is specified, warmup_ratio is ignored"})
    warmup_ratio: float = field(default=0)

    logging_steps: int = field(default=-1, metadata={"help": "When it is specified, logging_epoches is ignored"})
    logging_epoches: int = field(default=1)

    save_steps: int = field(default=-1, metadata={"help": "When it is specified, save_epoches is ignored"})
    save_epoches: int = field(default=1)
    save_total_limit: int = field(default=3)
    save_best: bool = field(default=False)

    seed: int = field(default=42)

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
    max_new_tokens: int = field(default=400)

@dataclass
class OutputArguments:
    logging_dir: str = field(default='wandb/')
    save_dir: str = field(default='checkpoints/')


def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, GenerationArguments, OutputArguments))
    model_args, data_args, training_args, generation_args, output_args = parser.parse_args_into_dataclasses()
    config_args_dict = model_args.__dict__.copy().update(dict(**data_args.__dict__, **training_args.__dict__))
    set_random_seed(training_args.seed)

    if data_args.dataset == 'gsm8k':
        from utils.gsm8k.datasets import make_finetuning_generator_data_module

    elif data_args.dataset == 'game24':
        from utils.game24.datasets import make_finetuning_generator_data_module

    else:
        raise NotImplementedError

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    # load model, tokenizer, and dataloader
    set_deepspeed_config(accelerator, training_args)
    model, tokenizer = build_model(model_args, training_args)
    data_module = make_finetuning_generator_data_module(tokenizer, data_args)
    train_dataloader, val_dataloader = make_training_dataloaders(data_module, training_args)
    
    # config optimizer and scheduler
    set_training_states(data_module, training_args)
    optimizer, lr_scheduler = get_optimizers(model, training_args)

    model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)

    cur_epoch = local_step = global_step = 0
    start_local_step = start_global_step = -1

    # init wandb
    if accelerator.is_main_process:
        project_name = os.environ['WANDB_PROJECT']
        logging_dir = os.path.join(output_args.logging_dir, project_name)

        os.makedirs(logging_dir, exist_ok=True)
        wandb_id = wandb.util.generate_id()
        wandb.init(id=wandb_id, dir=logging_dir, config=config_args_dict)


    # training
    global_step = 0
    model.train()
    while global_step < training_args.num_training_steps:
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Training - Epoch {cur_epoch+1} / {training_args.num_train_epoches}') if accelerator.is_main_process else enumerate(train_dataloader)
        
        for local_step, batch in train_dataloader_iterator:
            if global_step < start_global_step:
                global_step += 1
                continue

            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels')}
            # backpropagation
            with accelerator.accumulate(model):
                output = model(**batch_input, return_dict=True)
                loss = output.loss
                accelerator.backward(loss)

                optimizer.step()
                if not accelerator.optimizer_step_was_skipped and global_step % training_args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # training logging
            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=cur_epoch, step=local_step, loss=loss.item())

                if global_step % training_args.num_logging_steps == 0:
                    wandb.log({
                        'loss': loss.item(),
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)

            # save checkpoint
            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                save_llm_checkpoint(accelerator, model, tokenizer, output_args.save_dir, global_step, training_args.save_total_limit)

            # save states for resuming
            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(os.path.join(output_args.save_dir, 'resume'))

            global_step += 1

        cur_epoch += 1

        gc.collect(); torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    save_llm(accelerator, model, tokenizer, output_args.save_dir)
    save_training_args_with_accelerator(accelerator, training_args, output_args.save_dir)

    accelerator.save_state(os.path.join(output_args.save_dir, 'resume'))
    if accelerator.is_main_process:
        wandb.finish()

if __name__ == "__main__":
    main()
