from transformers import AdamW
from transformers import get_scheduler
import transformers
from typing import Optional, List, Dict, Set, Any, Union
from dataclasses import dataclass
import os

def get_optimizers(model: transformers.AutoModelForCausalLM, training_args: dataclass) -> Dict:
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optim = AdamW(
        optimizer_grouped_parameters, 
        lr=training_args.learning_rate, 
        # weight_decay=training_args.weight_decay
    )
    lr_scheduler = get_scheduler(
        training_args.lr_scheduler_type,
        optimizer=optim,
        # num_warmup_steps=training_args.num_updating_warmup_steps_aggr_devices,
        # num_training_steps=training_args.num_updating_steps_aggr_devices,
        num_warmup_steps=training_args.num_updating_warmup_steps,
        num_training_steps=training_args.num_updating_steps,
    )
    return optim, lr_scheduler



