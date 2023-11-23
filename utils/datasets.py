import json
import os
import re
import torch
import torch.nn.functional as F
from typing import Optional, Sequence, List, Set, Dict, Any, Union
import transformers
import logging
from dataclasses import dataclass
import pathlib
from torch.utils.data import DataLoader
from utils.constants import DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, IGNORE_INDEX



def read_jsonl(path: str):
    with open(path) as fh:
        return [json.loads(line) for line in fh.readlines() if line]


def get_few_shot_prompt(data_dir, prompt_file):
    with open(os.path.join(data_dir, prompt_file), 'r') as f:
        prompt = f.read()
    return prompt.replace('{', '{{').replace('}', '}}').replace('{{test_question}}', '{test_question}')


def get_model_solutions(data_dir, generator_id, target_set, process: bool = False):
    data_dir = os.path.join(data_dir, target_set)

    if process:
        files_pattern = f'responses_n*_{generator_id}_process.jsonl'
    else:
        files_pattern = f'responses_n*_{generator_id}.jsonl'

    response_files = [str(x) for x in pathlib.Path(data_dir).glob(files_pattern)]
    if not response_files:
        raise ValueError(f'Fail to find {files_pattern} in {data_dir}')

    ordering_and_response_path = []
    for response_file in response_files:
        regex_match = re.match(r".*responses_n([0-9]+)", response_file)
        if regex_match is not None:
            ordering_and_response_path.append((int(regex_match.group(1)), response_file))
    responses_sorted = sorted(ordering_and_response_path)
    responses_sorted = [response[1] for response in responses_sorted]
    read_file = responses_sorted[-1]

    examples = read_jsonl(read_file)
    print(f"{len(examples)} {target_set} examples, each with {len(examples[0]['outputs'])} solutions")
    return examples



def make_training_dataloaders(
    data_module: Dict[str, torch.utils.data.Dataset],
    training_args: dataclass = None,
) -> Dict:
    train_dataloader = DataLoader(
                            data_module['train_dataset'], 
                            batch_size=training_args.per_device_train_batch_size, 
                            shuffle=True, 
                            drop_last=False, 
                            collate_fn=data_module['train_dataset'].collate_fn, 
                        )
    if data_module['val_dataset'] is not None:
        val_dataloader = DataLoader(
                            data_module['val_dataset'], 
                            batch_size=training_args.per_device_eval_batch_size, 
                            shuffle=False, 
                            drop_last=False, 
                            collate_fn=data_module['val_dataset'].collate_fn, 
                        )
    else:
        val_dataloader = None
    return train_dataloader, val_dataloader



def make_testing_dataloader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, collate_fn=dataset.collate_fn)







def make_training_verifier_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    if data_args.process == True:
        dataset_class = ProcessVerifierDataset
    else:
        dataset_class = VerifierDataset

    train_dataset = dataset_class(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir, 
                        target_set=data_args.target_set, 
                        generator_id=data_args.generator_id, 
                        per_problem_sampling_solution=data_args.per_problem_sampling_solution, 
                        loss_level=data_args.loss_level,
                        loss_on_llm=data_args.loss_on_llm,
                        dedup=data_args.dedup,
                    )
    
    val_dataset = None
    if data_args.val_target_set is not None:
        val_dataset = dataset_class(
                            tokenizer=tokenizer, 
                            data_dir=data_args.data_dir, 
                            target_set=data_args.val_target_set, 
                            generator_id=data_args.generator_id, 
                            per_problem_sampling_solution=-1, 
                            loss_level=data_args.loss_level,
                            loss_on_llm=data_args.loss_on_llm,
                        )
    return dict(train_dataset=train_dataset, val_dataset=val_dataset)


def make_test_verifier_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: dataclass) -> Dict:
    test_dataset = VerifierDataset(
                        tokenizer=tokenizer, 
                        data_dir=data_args.data_dir,
                        target_set=data_args.target_set,
                        generator_id=data_args.generator_id, 
                        per_problem_sampling_solution=-1, 
                    )
    return test_dataset





class VerifierDataset(torch.utils.data.Dataset):
    """Right Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k/model_generation', 
        target_set: str = None,
        generator_id: str = None, 
        per_problem_sampling_solution: str = None, 
        loss_level: str = 'token', 
        loss_on_llm: bool = False,
        dedup: bool = False
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ('token', 'step')

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.examples = get_model_solutions(data_dir, generator_id, target_set)
        assert len(self.examples[0]['outputs']) >= per_problem_sampling_solution

        if per_problem_sampling_solution != -1:
            for example in self.examples:
                example['outputs'] = example['outputs'][:per_problem_sampling_solution]
        else:
            per_problem_sampling_solution = len(self.examples[0]['outputs'])
        

        if dedup:
            for ex in self.examples:
                dedup_outputs = []
                responses = set()
                for output in ex['outputs']:
                    if output['response'] in responses:
                        continue
                    responses.add(output['response'])
                    dedup_outputs.append(output)
                ex['outputs'] = dedup_outputs

        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)]
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [[outputs["response"] for outputs in ex["outputs"]] for ex in self.examples]
        v_classes = [[outputs["label"] == True for outputs in ex["outputs"]] for ex in self.examples]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        v_classes = self._flatten(v_classes)

        qns_tokens = tokenizer(qns_str, padding=False).input_ids
        solutions_tokens = tokenizer(solutions_str, padding=False, add_special_tokens=False).input_ids

        self.indices1 = indices1
        self.indices2 = indices2
        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.solutions_str = solutions_str
        self.solutions_tokens = solutions_tokens
        self.v_classes = v_classes

        self.n_question = len(self.examples)
        self.per_problem_sampling_solution = per_problem_sampling_solution

        print(f'Number of examples = {len(qns_str)} with #deduplication = {self.n_question * self.per_problem_sampling_solution - len(qns_str)}')

        self.max_len = max([
                len(self.qns_tokens[i]) + len(self.solutions_tokens[i]) + 1
                for i in range(len(self.solutions_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")
        
    def __len__(self):
        return len(self.solutions_tokens)

    def _flatten(self, ls):
        return [item for sublist in ls for item in sublist]

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        v_class = self.v_classes[idx]

        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]
        masks = (
            ([0] * len(qn_tokens))
            + ([1] * len(sol_tokens))
            + ([1])
        )

        # create language modeling labels
        if self.loss_on_llm:
            labels = input_ids
            labels = mask_labels(labels, masks)

        # create verifier labels
        if self.loss_level == 'token':
            v_labels = [int(v_class)] * len(input_ids)
            v_labels = mask_labels(v_labels, masks)
        else:
            raise NotImplementedError

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        v_labels = torch.tensor(v_labels)
        return dict(
            idx1=self.indices1[idx], idx2=self.indices2[idx], 
            input_ids=input_ids, labels=labels, v_labels=v_labels, 
            qn_str=self.qns_str[idx], qn_tokens=self.qns_tokens[idx], sol_str=self.solutions_str[idx], sol_tokens=self.solutions_tokens[idx], v_class=self.v_classes[idx],
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, v_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "v_labels"))
        idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class = tuple([instance[key] for instance in instances] for key in ("idx1", "idx2", "qn_str", "qn_tokens", "sol_str", "sol_tokens", "v_class"))

        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False) if self.loss_on_llm else None
        v_labels = right_pad_sequences(v_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            idx1=idx1, idx2=idx2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            v_labels=v_labels,
            qn_str=qn_str, qn_tokens=qn_tokens, sol_str=sol_str, sol_tokens=sol_tokens, v_class=v_class,
        )




class ProcessVerifierDataset(torch.utils.data.Dataset):
    """Right Padding"""
    def __init__(
        self, 
        tokenizer: transformers.PreTrainedTokenizer = None, 
        data_dir: str = 'data/gsm8k/model_generation', 
        target_set: str = None,
        generator_id: str = None, 
        per_problem_sampling_solution: str = None, 
        loss_level: str = 'token', 
        loss_on_llm: bool = False,
        dedup: bool = False
    ):
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.target_set = target_set
        self.generator_id = generator_id
        self.loss_level = loss_level
        self.loss_on_llm = loss_on_llm
        assert loss_level in ('token', 'step')

        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.examples = get_model_solutions(data_dir, generator_id, target_set, process=True)
        assert len(self.examples[0]['outputs']) >= per_problem_sampling_solution

        if per_problem_sampling_solution != -1:
            for example in self.examples:
                example['outputs'] = example['outputs'][:per_problem_sampling_solution]
        else:
            per_problem_sampling_solution = len(self.examples[0]['outputs'])
        

        if dedup:
            for ex in self.examples:
                dedup_outputs = []
                responses = set()
                for output in ex['outputs']:
                    if output['response'] in responses:
                        continue
                    responses.add(output['response'])
                    dedup_outputs.append(output)
                ex['outputs'] = dedup_outputs

        indices1 = [[i] * len(ex["outputs"]) for i, ex in enumerate(self.examples)]
        indices2 = [[j for j in range(len(ex["outputs"]))] for i, ex in enumerate(self.examples)]
        qns_str = [[ex["input"]] * len(ex["outputs"]) for ex in self.examples]
        solutions_str = [[outputs["response"] for outputs in ex["outputs"]] for ex in self.examples]
        step_labels = [[outputs["step_labels"] for outputs in ex["outputs"]] for ex in self.examples]
        v_classes = [[outputs["label"] == True for outputs in ex["outputs"]] for ex in self.examples]

        indices1 = self._flatten(indices1)
        indices2 = self._flatten(indices2)
        qns_str = self._flatten(qns_str)
        solutions_str = self._flatten(solutions_str)
        step_labels = self._flatten(step_labels)
        v_classes = self._flatten(v_classes)

        qns_tokens = tokenizer(qns_str, padding=False).input_ids

        steps_str = [
            list(map(lambda x: x+'\n', solution_str.split('\n')[:-1])) + [solution_str.split('\n')[-1]]
            for solution_str in solutions_str
        ]
        solutions_tokens = [
            [tokenizer.encode(step_str[0], add_special_tokens=False)]
              + [tokenizer.get_continued_input_ids(step) for step in step_str[1:]]
            for step_str in steps_str
        ]
        step_tokens_lens = [
            [len(step) for step in tokens]
            for tokens in solutions_tokens
        ]
        solutions_tokens = [self._flatten(tokens) for tokens in solutions_tokens]


        self.indices1 = indices1
        self.indices2 = indices2
        self.qns_str = qns_str
        self.qns_tokens = qns_tokens
        self.solutions_str = solutions_str
        self.solutions_tokens = solutions_tokens
        self.step_tokens_lens = step_tokens_lens
        self.step_labels = step_labels
        self.v_classes = v_classes

        self.n_question = len(self.examples)
        self.per_problem_sampling_solution = per_problem_sampling_solution

        print(f'Number of examples = {len(qns_str)} with #deduplication = {self.n_question * self.per_problem_sampling_solution - len(qns_str)}')

        self.max_len = max([
                len(self.qns_tokens[i]) + len(self.solutions_tokens[i]) + 1
                for i in range(len(self.solutions_tokens))
            ]
        )
        print(f"Max tokens: {self.max_len}")
        
    def __len__(self):
        return len(self.solutions_tokens)

    def _flatten(self, ls):
        return [item for sublist in ls for item in sublist]

    def __getitem__(self, idx):
        qn_tokens = self.qns_tokens[idx]
        sol_tokens = self.solutions_tokens[idx]
        step_labels = self.step_labels[idx]
        step_tokens_lens = self.step_tokens_lens[idx]

        input_ids = qn_tokens + sol_tokens + [self.eos_token_id]
        masks = (
            ([0] * len(qn_tokens))
            + ([1] * len(sol_tokens))
            + ([1])
        )

        # create language modeling labels
        if self.loss_on_llm:
            labels = input_ids
            labels = mask_labels(labels, masks)

        # create verifier labels
        if self.loss_level == 'token':
            v_labels = (
                [0] * len(qn_tokens) 
                + sum(
                    [
                        [1 if step_label else 0] * tokens_len
                        for tokens_len, step_label in zip(step_tokens_lens, step_labels)
                    ],
                    []
                ) 
                + [1 if step_labels[-1] else 0]
            )
            v_labels = mask_labels(v_labels, masks)

            assert len(v_labels) == len(input_ids)
        else:
            raise NotImplementedError

        input_ids = torch.tensor(input_ids)
        labels = torch.tensor(labels) if self.loss_on_llm else None
        v_labels = torch.tensor(v_labels)
        return dict(
            idx1=self.indices1[idx], idx2=self.indices2[idx], 
            input_ids=input_ids, labels=labels, v_labels=v_labels, 
            qn_str=self.qns_str[idx], qn_tokens=self.qns_tokens[idx], sol_str=self.solutions_str[idx], sol_tokens=self.solutions_tokens[idx], v_class=self.v_classes[idx],
        )

    def collate_fn(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, v_labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "v_labels"))
        idx1, idx2, qn_str, qn_tokens, sol_str, sol_tokens, v_class = tuple([instance[key] for instance in instances] for key in ("idx1", "idx2", "qn_str", "qn_tokens", "sol_str", "sol_tokens", "v_class"))

        input_ids, attention_mask = right_pad_sequences(input_ids, padding_value=self.pad_token_id, return_attention_mask=True)
        labels = right_pad_sequences(labels, padding_value=IGNORE_INDEX, return_attention_mask=False) if self.loss_on_llm else None
        v_labels = right_pad_sequences(v_labels, padding_value=IGNORE_INDEX, return_attention_mask=False)
        
        return dict(
            idx1=idx1, idx2=idx2,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            v_labels=v_labels,
            qn_str=qn_str, qn_tokens=qn_tokens, sol_str=sol_str, sol_tokens=sol_tokens, v_class=v_class,
        )





def left_pad_sequences(sequences: List[torch.LongTensor], padding_value: int, return_attention_mask: bool = False):
    max_length = max(len(x) for x in sequences)
    padded_sequences = torch.stack([F.pad(seq, (max_length - seq.shape[-1], 0), value=padding_value) for seq in sequences], dim=0)
    if return_attention_mask:
        attention_mask = padded_sequences.ne(padding_value)
        return padded_sequences, attention_mask
    return padded_sequences

def right_pad_sequences(sequences: List[torch.LongTensor], padding_value: int, return_attention_mask: bool = False):
    padded_sequences = torch.nn.utils.rnn.pad_sequence(
        sequences,
        batch_first=True,
        padding_value=padding_value,
    )
    if return_attention_mask:
        attention_mask = padded_sequences.ne(padding_value)
        return padded_sequences, attention_mask
    return padded_sequences


def mask_labels(labels: List[int], masks: List[bool]):
    """Mask the corresponding label into IGNORE_INDEX"""
    assert len(labels) == len(masks)
    return [
        token if mask
        else IGNORE_INDEX
        for token, mask in zip(labels, masks) 
    ]



