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
import shutil

from utils.states import set_deepspeed_config, set_training_states, set_random_seed
from utils.optim import get_optimizers
from utils.models import save_training_args_with_accelerator
from utils.verifier_models import save_verifier, save_verifier_checkpoint, save_best_verifier_checkpoint, build_verifier
from utils.datasets import make_training_verifier_data_module, make_training_dataloaders
from utils.metrics import VerifierClassificationAcc


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")

@dataclass
class DataArguments:
    data_dir: str = field(default='data/gsm8k/model_generation', metadata={"help": "Path to the training data."})
    target_set: str = field(default='train')
    val_target_set: str = field(default=None)
    generator_id: str = field(default='llama7b-2-ep2')

    per_problem_sampling_solution: int = field(default=-1)
    loss_level: str = field(default='token')
    loss_on_llm: bool = field(default=False)

    dedup: bool = field(default=False)
    process: bool = field(default=False)

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

    num_lr_epoches_fs: int = field(default=-1)
    num_lr_epoches_scatter: int = field(default=-1)

    logging_steps: int = field(default=-1, metadata={"help": "When it is specified, logging_epoches is ignored"})
    logging_epoches: int = field(default=1)

    save_steps: int = field(default=-1, metadata={"help": "When it is specified, save_epoches is ignored"})
    save_epoches: int = field(default=1)
    save_total_limit: int = field(default=3)
    save_best: bool = field(default=False)

    seed: int = field(default=42)
    resume: bool = field(default=False)

@dataclass
class OutputArguments:
    logging_dir: str = field(default='wandb/')
    save_dir: str = field(default='checkpoints/')



def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, OutputArguments))
    model_args, data_args, training_args, output_args = parser.parse_args_into_dataclasses()
    config_args_dict = model_args.__dict__.copy().update(dict(**data_args.__dict__, **training_args.__dict__))
    set_random_seed(training_args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    # load model, tokenizer, and dataloader
    set_deepspeed_config(accelerator, training_args)
    model, tokenizer = build_verifier(model_args, training_args)
    data_module = make_training_verifier_data_module(tokenizer, data_args)
    train_dataloader, val_dataloader = make_training_dataloaders(data_module, training_args)
    
    # config optimizer and scheduler
    set_training_states(data_module, training_args)
    optimizer, lr_scheduler = get_optimizers(model, training_args)

    # init validation metric
    val_metric = VerifierClassificationAcc(n_data=len(data_module['val_dataset']) if data_module['val_dataset'] is not None else 0)


    if val_dataloader is not None:
        model, train_dataloader, val_dataloader, optimizer = accelerator.prepare(model, train_dataloader, val_dataloader, optimizer)
    else:
        model, train_dataloader, optimizer = accelerator.prepare(model, train_dataloader, optimizer)


    cur_epoch = local_step = global_step = 0
    start_local_step = start_global_step = -1
    best_val_acc = 0


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
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc='Training') if accelerator.is_main_process else enumerate(train_dataloader)
        
        for local_step, batch in train_dataloader_iterator:
            if global_step < start_global_step:
                global_step += 1
                continue
            
            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels')}
            # backpropagation
            with accelerator.accumulate(model):
                output = model(**batch_input, output_all_losses=True)
                loss = output.loss
                all_losses = output.all_losses
                accelerator.backward(loss)

                optimizer.step()
                if not accelerator.optimizer_step_was_skipped and global_step % training_args.gradient_accumulation_steps == 0:
                    lr_scheduler.step()
                optimizer.zero_grad()

            # training logging
            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=cur_epoch, step=local_step, loss=loss.item(), v_loss=all_losses.get('v_loss').item(), llm_loss=all_losses.get('llm_loss').item() if data_args.loss_on_llm else 0)

                if global_step % training_args.num_logging_steps == 0:
                    wandb.log({
                        'loss': loss.item(),
                        'v_loss': all_losses.get('v_loss').item(),
                        'llm_loss': all_losses.get('llm_loss').item() if data_args.loss_on_llm else 0,
                        'lr': lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

            # save checkpoint
            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                save_verifier_checkpoint(accelerator, model, tokenizer, output_args.save_dir, global_step, training_args.save_total_limit)

            # evaluation
            if val_dataloader is not None and \
                global_step != 0 and (global_step % training_args.per_eval_steps == 0 or global_step == training_args.num_training_steps - 1):
                
                gc.collect(); torch.cuda.empty_cache()
                model.eval()

                ## generate
                val_dataloader_iterator = tqdm(enumerate(val_dataloader), total=len(val_dataloader), desc='Evaluation') if accelerator.is_main_process else enumerate(val_dataloader)
                for _, eval_batch in val_dataloader_iterator:
                    batch_input = {k: v for k, v in eval_batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels')}
                    with torch.inference_mode(mode=True):
                        output = model(**batch_input, output_all_losses=True)
                        loss = output.loss
                        v_scores = output.v_scores
                        all_losses = output.all_losses

                    val_metric(v_scores, eval_batch['v_labels'])

                ## validation logging
                if accelerator.is_main_process:
                    val_loss = loss.item()
                    val_v_loss = all_losses.get('v_loss').item()
                    val_llm_loss = all_losses.get('llm_loss').item() if data_args.loss_on_llm else 0
                    val_acc = val_metric.get_metric()
                    wandb.log({
                        'val_loss': val_loss,
                        'val_v_loss': val_v_loss,
                        'val_llm_loss': val_llm_loss if data_args.loss_on_llm else 0,
                        'val_acc': val_acc,
                    }, step=global_step)
                    accelerator.print(f"Epoch: {cur_epoch}, Step: {local_step}, Val loss: {val_loss}, Val v_loss: {val_v_loss}, Val llm_loss: {val_llm_loss}, Val acc: {val_acc}")

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        accelerator.print(f"Current best! val_acc={best_val_acc}")
                        if training_args.save_best:
                            save_best_verifier_checkpoint(accelerator, model, tokenizer, output_args.save_dir, global_step, training_args.save_total_limit)

                gc.collect(); torch.cuda.empty_cache()
                model.train()

            # save states for resuming
            if global_step != 0 and global_step % training_args.per_save_steps == 0:
                accelerator.wait_for_everyone()
                accelerator.save_state(os.path.join(output_args.save_dir, 'resume'))


            global_step += 1

        cur_epoch += 1
        if cur_epoch == 1:
            accelerator.wait_for_everyone()
            save_verifier_checkpoint(accelerator, model, tokenizer, output_args.save_dir, global_step, training_args.save_total_limit)

        del train_dataloader_iterator
        gc.collect(); torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    save_verifier(accelerator, model, tokenizer, output_args.save_dir)
    save_training_args_with_accelerator(accelerator, training_args, output_args.save_dir)

    if accelerator.is_main_process:
        shutil.rmtree(os.path.join(output_args.save_dir, 'resume'))
        wandb.finish()


if __name__ == "__main__":
    main()
