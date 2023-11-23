"""
Batch Generation with Calculator using Cache
"""



import torch
import json
import os
import re
import transformers
from transformers import StoppingCriteria, StoppingCriteriaList, LogitsProcessor, LogitsProcessorList
import torch.nn.functional as F
import torch.nn as nn
from accelerate.accelerator import Accelerator
from typing import Union, Sequence, List, Set, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import math
import gc
from tqdm import tqdm
from transformers.generation.utils import ModelOutput
from utils.cached_models import PreTrainedTokenizer
from utils.constants import LLAMA_EQUALS_TOKENS, LLAMA_LEFTMARK_TOKENS, LLAMA_RIGHTMARK_TOKEN, LLAMA_NEWLINE_TOKEN
from utils.gsm8k.decoding import use_calculator



@dataclass
class SamplingOutput(ModelOutput):
    sequences: torch.LongTensor = None
    transition_scores: Optional[torch.FloatTensor] = None
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None

@dataclass
class StepSamplingOutput(ModelOutput):
    sequences: torch.LongTensor = None
    steps: torch.LongTensor = None
    transition_scores: Optional[torch.FloatTensor] = None
    verifier_scores: Optional[torch.FloatTensor] = None
    past_key_values: Tuple[Tuple[torch.FloatTensor]] = None


class BatchCalculatorCallingCriteria(StoppingCriteria):
    # stop when one sample in the batch meets the calculator marker
    def __init__(self, keywords_ids: set, left_mark_tokens: set, device: torch.device):
        self.keyword_tokens = torch.tensor(list(keywords_ids)).to(device)
        self.left_mark_tokens = torch.tensor(list(left_mark_tokens)).to(device)

    def _is_hit_keywords(self, input_ids: torch.LongTensor) -> bool:
        return input_ids.unsqueeze(-1).eq(self.keyword_tokens.view((1, 1, -1))).any(2)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        is_hit_keywords = self._is_hit_keywords(input_ids[:, -1:])
        if not is_hit_keywords.any():
            return False

        index = is_hit_keywords.nonzero()[:, 0]
        token_ids = input_ids.index_select(0, index)[:, :-1]

        last_mark_indices = find_rightmost_tokens_positions(token_ids, self.left_mark_tokens, wnby=False)
        is_hit_left_mark = last_mark_indices.ne(-1)
        is_hit_keywords_after_mark = count_tokens_after_positions(token_ids, tokens=self.keyword_tokens, positions=last_mark_indices, include_pos=False).ne(0)
        if torch.logical_and(is_hit_left_mark, ~is_hit_keywords_after_mark).any():
            return True
        return False

class BatchEndStoppingCriteria(StoppingCriteria):
    # stop when all samples in the batch have generated the end token
    def __init__(self, end_token_id: int, device: torch.device):
        self.end_token_id = torch.tensor([end_token_id]).to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids.eq(self.end_token_id).any(1).all():
            return True
        return False

class BatchMultipleEndStoppingCriteria(StoppingCriteria):
    # stop when all samples in the batch have generated the end token. allow specify multiple end tokens
    def __init__(self, end_token_ids: Set[int], device: torch.device):
        self.end_token_ids = torch.tensor(list(end_token_ids)).to(device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if input_ids[:, :, None].eq(self.end_token_ids.view((1, 1, -1))).any(2).any(1).all():
            return True
        return False

class StepStoppingCriteria(StoppingCriteria):
    # stop when all samples in the batch have completed one step
    def __init__(self, cur_token_lens: torch.LongTensor, end_token_ids: Set[int], pad_token_id: int, device: torch.device):
        self.cur_token_lens = cur_token_lens
        self.end_token_ids = torch.tensor(list(end_token_ids)).to(device)
        self.pad_token_id = pad_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        new_tokens = get_new_generated_tokens(input_ids, past_token_lens=self.cur_token_lens, pad_token_id=self.pad_token_id)
        if new_tokens[:, :, None].eq(self.end_token_ids.view((1, 1, -1))).any(2).any(1).all():
            return True
        return False




def find_leftmost_tokens_positions(input_ids: torch.LongTensor, tokens: Union[int, torch.LongTensor], wnby: bool=True) -> torch.LongTensor:
    """
    Get the indices where `tokens` first appear in the `input_ids` for each sample in the batch. When there aren't `tokens`, return seq_len-1 when `within_boundary`

    e.g.
    input_ids = torch.tensor([[1, 2, 3, 3], [7, 0, 4, 0], [3, 2, 1, 2]])
    tokens = torch.tensor([3, 0])
    find_leftmost_tokens_positions(input_ids, tokens)
    >> tensor([2, 1, 0])

    tokens = torch.tensor([3, 2])
    find_leftmost_tokens_positions(input_ids, tokens, wnby=True)
    >> tensor([1, 3, 0])

    find_leftmost_tokens_positions(input_ids, tokens, wnby=False)
    >> tensor([1, 4, 0])
    """
    assert input_ids.ndim == 2
    bsz, seq_len = input_ids.shape
    if isinstance(tokens, int):
        mask = input_ids.eq(tokens)
    elif isinstance(tokens, torch.Tensor):
        mask = input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2)
    positions = torch.where(mask.any(1), mask.float().argmax(dim=1), seq_len-1 if wnby else seq_len)
    return positions


def find_rightmost_tokens_positions(input_ids: torch.LongTensor, tokens: Union[int, torch.LongTensor], wnby: bool=True) -> torch.LongTensor:
    """
    Get the index where `tokens` last appear in the `input_ids` for each sample in the batch. When there aren't `tokens`, return 0 when `within_boundary`

    e.g.
    input_ids = torch.tensor([[1, 2, 3, 3], [7, 0, 4, 0], [3, 2, 1, 2]])
    tokens = torch.tensor([3, 0])
    find_rightmost_tokens_positions(input_ids, tokens)
    >> tensor([3, 3, 0])

    tokens = torch.tensor([3, 2])
    find_rightmost_tokens_positions(input_ids, tokens, wnby=True)
    >> tensor([3, 0, 3])

    find_rightmost_tokens_positions(input_ids, tokens, wnby=False)
    >> tensor([3, -1, 3])
    """
    assert input_ids.ndim == 2
    bsz, seq_len = input_ids.shape
    if isinstance(tokens, int):
        mask = input_ids.eq(tokens)
    elif isinstance(tokens, torch.Tensor):
        mask = input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2)
    positions = torch.where(mask.any(1), (seq_len - 1) - mask.flip(dims=[1]).float().argmax(dim=1), 0 if wnby else -1)
    return positions


def find_leftmost_notpadded_positions(tensor: torch.Tensor, pad_value: Union[int, float], wnby: bool=True) -> torch.Tensor:
    """Get the index of the first not-pad token in the left for each sample in the batch `tensor`. When they are all pad_value, return seq_len-1 when within_boundary"""
    assert tensor.ndim == 2
    bsz, seq_len = tensor.shape
    mask = tensor.ne(pad_value)
    positions = torch.where(mask.any(1), mask.float().argmax(dim=1), seq_len-1 if wnby else seq_len)
    return positions


def find_rightmost_notpadded_positions(tensor: torch.Tensor, pad_value: Union[int, float], wnby: bool=True) -> torch.Tensor:
    """For right padding. Get the index of the last not-pad token for each sample in the batch `tensor`. When they are all pad_value, return 0 when within_boundary"""
    assert tensor.ndim == 2
    bsz, seq_len = tensor.shape
    mask = tensor.ne(pad_value)
    positions = torch.where(mask.any(1), (seq_len - 1) - mask.flip(dims=[1]).float().argmax(dim=1), 0 if wnby else -1)
    return positions


def count_right_padding(tensor: torch.Tensor, pad_value: Union[int, float]) -> torch.Tensor:
    """For right padding. Count pad_value in the right of `tensor`"""
    seq_len = tensor.shape[-1]
    positions = find_rightmost_notpadded_positions(tensor, pad_value=pad_value, wnby=False)
    return (seq_len - 1) - positions


def count_left_padding(tensor: torch.Tensor, pad_value: Union[int, float]) -> torch.Tensor:
    """For left padding. Count pad_value in the left of `tensor`"""
    seq_len = tensor.shape[-1]
    positions = find_leftmost_notpadded_positions(tensor, pad_value=pad_value, wnby=False)
    return positions


def count_not_left_padding(tensor: torch.Tensor, pad_value: Union[int, float]) -> torch.Tensor:
    """For left padding. Count not pad_value of `tensor`"""
    counts = count_left_padding(tensor, pad_value=pad_value)
    return tensor.shape[-1] - counts


def count_shared_left_padding(tensor: torch.Tensor, pad_value: Union[int, float]) -> torch.Tensor:
    """For left padding. Return the minimal padding length in the batch `tensor`"""
    return count_left_padding(tensor, pad_value).min()




def get_mask_for_seq_area(tensor: torch.Tensor, left_borders: Optional[torch.LongTensor]=None, right_borders: Optional[torch.LongTensor]=None, include_left: bool=False, include_right: bool=False):
    """Return a mask with True in the specified areas"""
    assert not (left_borders is None and right_borders is None)
    bsz, seq_len = tensor.shape

    if include_left and left_borders is not None:
        left_borders = left_borders - 1
    if include_right and right_borders is not None:
        right_borders = right_borders + 1

    if left_borders is not None and right_borders is not None:
        mask = torch.logical_and(
            torch.arange(seq_len).view(1, -1).to(tensor.device) > left_borders.view(-1, 1),
            torch.arange(seq_len).view(1, -1).to(tensor.device) < right_borders.view(-1, 1)
        )
    elif left_borders is not None:
        mask = (torch.arange(seq_len).view(1, -1).to(tensor.device) > left_borders.view(-1, 1))
    elif right_borders is not None:
        mask = (torch.arange(seq_len).view(1, -1).to(tensor.device) < right_borders.view(-1, 1))
    return mask


def mask_by_borders_2D(
    tensor: torch.Tensor, 
    left_borders: Optional[torch.LongTensor] = None, 
    right_borders: Optional[torch.LongTensor] = None, 
    include_left: bool = False, 
    include_right: bool = False,
    value: Union[int, float] = 0,
):
    """Fill before/after borders into value"""
    mask = get_mask_for_seq_area(tensor=tensor, left_borders=left_borders, right_borders=right_borders, include_left=include_left, include_right=include_right)
    return tensor.masked_fill(mask, value=value)


def mask_by_borders_past_key_values(
    past_key_values: Tuple[Tuple[torch.FloatTensor]], 
    left_borders: torch.LongTensor = None, 
    right_borders: torch.LongTensor = None, 
    include_left: bool = False, 
    include_right: bool = False,
    value: Union[int, float] = 0,
):
    """Fill before/after borders into value"""
    mask = get_mask_for_seq_area(past_key_values[0][0][:, 0, :, 0], left_borders=left_borders, right_borders=right_borders, include_left=include_left, include_right=include_right)
    mask = mask[:, None, :, None].expand_as(past_key_values[0][0])

    return tuple(tuple(past_key_value.masked_fill(mask, value=value) for past_key_value in layer_past_key_values) for layer_past_key_values in past_key_values)



def count_tokens_after_positions(input_ids: torch.LongTensor, positions: torch.LongTensor, tokens: Union[int, torch.LongTensor], include_pos: bool=False) -> torch.LongTensor:
    """Count `tokens` after `positions`"""
    seq_len = input_ids.shape[-1]
    mask = get_mask_for_seq_area(input_ids, right_borders=positions, include_right=not include_pos)
    input_ids = input_ids.masked_fill(mask, value=-1)
    if isinstance(tokens, int):
        return input_ids.eq(tokens).sum(1)
    elif isinstance(tokens, torch.Tensor):
        return input_ids[:, :, None].eq(tokens.view(1, 1, -1)).any(2).sum(1)
    

def get_new_generated_tokens(input_ids: torch.LongTensor, past_token_lens: torch.LongTensor, pad_token_id: int=0):
    """Mask past tokens and only reserve the newly generated tokens"""
    n_paddings = count_left_padding(input_ids, pad_value=pad_token_id)
    return mask_by_borders_2D(input_ids, right_borders=n_paddings + past_token_lens, include_right=False, value=pad_token_id)

    

def batched_shift_along_seq_dim_2D(tensor: torch.Tensor, shifts: torch.LongTensor=None):
    """Shift a tensor based on the shifts along seq_dim"""
    bsz, seq_len = tensor.shape
    assert shifts.numel() == bsz

    arange1 = torch.arange(seq_len).view((1, seq_len)).to(tensor.device)
    arange2 = ((arange1 - shifts.view((bsz, 1))) % seq_len)

    return torch.gather(tensor, 1, arange2)



def batched_shift_along_seq_dim_past_key_values(past_key_values: Tuple[Tuple[torch.FloatTensor]], shifts: torch.LongTensor=None):
    """Shift a tensor based on the shifts along seq_dim"""
    bsz = past_key_values[0][0].shape[0]
    seq_len = past_key_values[0][0].shape[2]
    assert shifts.numel() == bsz

    arange1 = torch.arange(seq_len).view((1, seq_len)).to(past_key_values[0][0].device)
    arange2 = ((arange1 - shifts.view((bsz, 1))) % seq_len)

    arange2 = arange2[:, None, :, None].expand_as(past_key_values[0][0])
    return tuple(tuple(torch.gather(past_key_values[i][j], 2, arange2) for j in range(len(past_key_values[i]))) for i in range(len(past_key_values)))



def shift_padding_to_left_2D(tensor: torch.Tensor, pad_value: Union[int, float] = 0):
    """Shift right padding in `tensor` to the left"""
    bsz, seq_len = tensor.shape
    shifts = count_right_padding(tensor, pad_value=pad_value)

    return batched_shift_along_seq_dim_2D(tensor, shifts=shifts)


def shift_padding_to_right_2D(tensor: torch.Tensor, pad_value: Union[int, float] = 0):
    """Shift left padding in `tensor` to the right"""
    bsz, seq_len = tensor.shape
    shifts = count_left_padding(tensor, pad_value=pad_value)

    return batched_shift_along_seq_dim_2D(tensor, shifts=-shifts)





class SamplingWithCalculator:
    def __init__(
        self,
        accelerator: Accelerator = None,
        model: transformers.PreTrainedModel = None, 
        verifier: nn.Module = None,
        tokenizer: PreTrainedTokenizer = None, 
        generation_args: dataclass = None,
    ):
        self.accelerator = accelerator
        self.model = model
        self.verifier = verifier
        self.tokenizer = tokenizer
        self.generation_args = generation_args
        self.device = accelerator.device

        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.equal_token_ids = torch.LongTensor(list(LLAMA_EQUALS_TOKENS)).to(self.device)
        self.left_mark_token_ids = torch.LongTensor(list(LLAMA_LEFTMARK_TOKENS)).to(self.device)
        self.right_mark_token_ids = torch.LongTensor([LLAMA_RIGHTMARK_TOKEN]).to(self.device)
        self.newline_token_ids = torch.LongTensor([LLAMA_NEWLINE_TOKEN]).to(self.device)
        self.inter_step_end_token_ids = self.newline_token_ids
        self.step_end_token_ids = torch.concat([self.newline_token_ids, torch.tensor([self.eos_token_id], device=self.device)])

        self.max_new_tokens = generation_args.max_new_tokens
        self.max_length = generation_args.max_length
        self.generation_config = {k:v for k,v in generation_args.__dict__.items() if k not in ('max_new_tokens', 'max_length')}

    def _shift_padding_to_left(self, token_ids: torch.LongTensor, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, transition_scores: torch.FloatTensor=None):
        """Shift right padding in `token_ids` to the left, and adjust `past_key_values` and `transition_scores` correspondingly"""
        bsz, seq_len = token_ids.shape
        shifts = count_right_padding(token_ids, pad_value=self.pad_token_id)

        token_ids = batched_shift_along_seq_dim_2D(token_ids, shifts=shifts)

        past_key_values = batched_shift_along_seq_dim_past_key_values(past_key_values, shifts=shifts) if past_key_values is not None else None
        transition_scores = shift_padding_to_left_2D(transition_scores, pad_value=0) if transition_scores is not None else None
        return token_ids, past_key_values, transition_scores

    def _truncate_left_padding(self, token_ids: torch.LongTensor, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, transition_scores: torch.FloatTensor=None):
        n_truncate = count_shared_left_padding(token_ids, pad_value=self.pad_token_id)

        token_ids = token_ids[:, n_truncate:]
        if past_key_values is not None:
            past_key_values = tuple(tuple(past_key_value[:, :, n_truncate:] for past_key_value in layer_past_key_values) for layer_past_key_values in past_key_values)

        if transition_scores is not None:
            n_truncate = count_shared_left_padding(transition_scores, pad_value=0)
            transition_scores = transition_scores[:, n_truncate:]
        return token_ids, past_key_values, transition_scores

    def _add_new_tokens_and_adjust(self, token_ids: torch.LongTensor, new_ids: torch.LongTensor, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, transition_scores: torch.FloatTensor=None):
        """Add the new token ids to the existing batch and adjust the past_key_values and transition_scores accordingly. Shift as left padding + truncate redundant ones"""
        new_token_ids = torch.cat([token_ids, new_ids], dim=1)
        new_past_key_values = past_key_values
        new_transition_scores = transition_scores

        shifts = count_right_padding(new_token_ids, pad_value=self.pad_token_id)
        new_token_ids = batched_shift_along_seq_dim_2D(new_token_ids, shifts=shifts)
        if past_key_values is not None:
            new_past_key_values = batched_shift_along_seq_dim_past_key_values(new_past_key_values, shifts=shifts)
            new_past_key_values = mask_by_borders_past_key_values(new_past_key_values, right_borders=shifts, include_right=False, value=0)
        if transition_scores is not None:
            new_scores = torch.zeros_like(new_ids).to(transition_scores.dtype)
            new_scores.masked_fill_(new_ids.ne(self.pad_token_id), 100)
            new_transition_scores = torch.concat([transition_scores, new_scores], dim=1)
            new_transition_scores = shift_padding_to_left_2D(new_transition_scores, pad_value=0)

        new_token_ids, new_past_key_values, new_transition_scores = self._truncate_left_padding(new_token_ids, new_past_key_values, new_transition_scores)
        return new_token_ids, new_past_key_values, new_transition_scores

    def _call_calculator(self, token_ids: torch.FloatTensor) -> List[str]:
        """Identify which samples require to use calculator and call calculator for them. Return correspondingly new token_ids"""
        indices = token_ids[:, -1, None].eq(self.equal_token_ids.view(1, -1)).any(1)
        indices = torch.arange(token_ids.shape[0])[indices.cpu()]

        new_text_list = [''] * token_ids.shape[0]
        for i in indices:
            answer = use_calculator(self.tokenizer.decode(token_ids[i]))
            if answer is not None:
                new_text_list[i] += (str(answer) + ">>")

        new_token_ids = self.tokenizer.get_continued_input_ids(new_text_list, right_padding=True, return_tensors=True).to(self.device)
        return new_token_ids

    def _cal_generation_scores(self, transition_scores: torch.FloatTensor):
        length = (transition_scores < 0).sum(1)
        length_penalty = self.generation_config.get('length_penalty', 1)
        return transition_scores.clip(max=0).sum(1) / (length**length_penalty)

    @torch.inference_mode(mode=True)
    def _sample_tokens_with_calculator(
        self, 
        input_ids: torch.LongTensor = None, 
        past_key_values: Tuple[Tuple[torch.FloatTensor]] = None, 
        max_length: int = None, 
        stopping_criteria_end: StoppingCriteria = None, 
        logits_processor: LogitsProcessorList = None, 
        output_transition_scores: bool = False,
        output_past_key_values: bool = True,
        **kwargs,
    ) -> SamplingOutput:
        """
        Batch sampling with calculator - model generation (token-level)
        """

        # if (
        #     self.pad_token_id is not None
        #     and len(input_ids.shape) == 2
        #     and (input_ids[:, 0] == self.pad_token_id).all()
        # ):
        #     print(
        #         "There are extra padding in the left!"
        #     )

        if stopping_criteria_end is None:
            stopping_criteria_end = BatchEndStoppingCriteria(end_token_id=self.eos_token_id, device=self.device)
        stopping_criteria_calculator = BatchCalculatorCallingCriteria(keywords_ids=LLAMA_EQUALS_TOKENS, left_mark_tokens=LLAMA_LEFTMARK_TOKENS, device=self.device)
        stopping_criteria = StoppingCriteriaList([stopping_criteria_calculator, stopping_criteria_end])

        if logits_processor is None:
            logits_processor = LogitsProcessorList()

        all_transition_scores = None
        if output_transition_scores:
            all_transition_scores = torch.empty((input_ids.shape[0], 0)).float().to(self.device)

        input_token_lens = count_not_left_padding(input_ids, pad_value=self.pad_token_id)

        cur_length = input_ids.shape[-1]
        while cur_length < max_length:
            max_new_tokens = max_length - cur_length
            outputs = self.accelerator.unwrap_model(self.model).generate(
                input_ids=input_ids, 
                attention_mask=input_ids.ne(self.pad_token_id),
                past_key_values=past_key_values,
                max_new_tokens=max_new_tokens, 
                stopping_criteria=stopping_criteria,
                logits_processor=logits_processor,
                output_scores=output_transition_scores,
                return_dict_in_generate=True,
                **kwargs,
                **self.generation_config,
            )
            input_ids, past_key_values = outputs.sequences, outputs.past_key_values
            if output_transition_scores:
                transition_scores = self.accelerator.unwrap_model(self.model).compute_transition_scores(sequences=input_ids, scores=outputs.scores, beam_indices=outputs.get('beam_indices'), normalize_logits=True)
                all_transition_scores = torch.cat([all_transition_scores, transition_scores], dim=1)

            if stopping_criteria_end(input_ids, None):
                break
            
            new_token_ids = self._call_calculator(input_ids)
            if new_token_ids.numel():
                input_ids, past_key_values, all_transition_scores = self._add_new_tokens_and_adjust(input_ids, new_token_ids, past_key_values, all_transition_scores)
                
            cur_length = input_ids.shape[-1]

        # For the convenience of subsequent operations/processing
        input_ids, past_key_values, all_transition_scores = self._cut_after_eos_lp(input_ids, past_key_values, all_transition_scores, past_token_lens=input_token_lens)

        return SamplingOutput(
            sequences=input_ids,
            transition_scores=all_transition_scores,
            past_key_values=past_key_values if output_past_key_values else None,
        )


    def _convert_into_tensors(self, qns: Union[str, List[str], torch.LongTensor]):
        if isinstance(qns, list) and isinstance(qns[0], str):
            token_ids = self.tokenizer(qns, padding=True, return_tensors='pt').input_ids
        elif isinstance(qns, str):
            token_ids = self.tokenizer([qns], return_tensors='pt').input_ids
        elif isinstance(qns, torch.Tensor):
            token_ids = qns
            if token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(0)
        else:
            raise ValueError

        return token_ids.to(self.device)

    def sample(
        self, 
        qns: Union[str, List[str]],
    ) -> Union[str, List[str]]:
        """
        Batch sampling with calculator (string-level)

        Return:
            responses (`Union[str, List[str]]`)
        """
        
        input_ids = self._convert_into_tensors(qns)

        cur_length = input_ids.shape[-1]
        if self.max_new_tokens > 0:
            max_length = cur_length + self.max_new_tokens
        else:
            max_length = self.max_length
        
        outputs = self._sample_tokens_with_calculator(
            input_ids=input_ids,
            max_length=max_length,
            stopping_criteria_end=BatchEndStoppingCriteria(end_token_id=self.eos_token_id, device=self.device),
        )
        completions = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        responses = [completion[len(qn):].strip() for qn, completion in zip(qns, completions)]
        
        if isinstance(qns, str):
            return responses[0]
        return responses

    def _cut_after_eos_lp(self, input_ids: torch.LongTensor, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, transition_scores: torch.FloatTensor=None, past_token_lens: torch.LongTensor=None):
        """Mask the tokens after eos and keep it left padding"""
        new_past_key_values = past_key_values
        new_transition_scores = transition_scores

        valid_borders_right = find_leftmost_tokens_positions(input_ids, self.eos_token_id, wnby=True)

        new_input_ids = mask_by_borders_2D(input_ids, left_borders=valid_borders_right, include_left=False, value=self.pad_token_id)

        if past_key_values is not None:
            new_past_key_values = mask_by_borders_past_key_values(past_key_values, left_borders=valid_borders_right, include_left=False, value=0)
        
        if transition_scores is not None:
            generate_begin_indices = count_left_padding(input_ids, pad_value=self.pad_token_id) + past_token_lens
            n_left_padding = count_left_padding(transition_scores, pad_value=0)
            borders_for_transitions = valid_borders_right - generate_begin_indices + n_left_padding
            new_transition_scores = mask_by_borders_2D(transition_scores, left_borders=borders_for_transitions, include_left=False, value=0)

        new_input_ids, new_past_key_values, new_transition_scores = self._shift_padding_to_left(new_input_ids, new_past_key_values, new_transition_scores)
        return new_input_ids, new_past_key_values, new_transition_scores

    def _cut_latter_steps(self, input_ids: torch.LongTensor, past_key_values: Tuple[Tuple[torch.FloatTensor]]=None, transition_scores: torch.FloatTensor=None, past_token_lens: torch.LongTensor=None):
        """Mask the latter steps and keep it left padding"""
        new_past_key_values = past_key_values
        new_transition_scores = transition_scores

        new_tokens = get_new_generated_tokens(input_ids, past_token_lens=past_token_lens, pad_token_id=self.pad_token_id)
        cur_step_borders_right = find_leftmost_tokens_positions(new_tokens, self.step_end_token_ids, wnby=True)

        new_input_ids = mask_by_borders_2D(input_ids, left_borders=cur_step_borders_right, include_left=False, value=self.pad_token_id)

        if past_key_values is not None:
            new_past_key_values = mask_by_borders_past_key_values(past_key_values, left_borders=cur_step_borders_right, include_left=False, value=0)

        if transition_scores is not None:
            generate_begin_indices = count_left_padding(input_ids, pad_value=self.pad_token_id) + past_token_lens
            n_left_padding = count_left_padding(transition_scores, pad_value=0)
            borders_for_transitions = cur_step_borders_right - generate_begin_indices + n_left_padding
            new_transition_scores = mask_by_borders_2D(transition_scores, left_borders=borders_for_transitions, include_left=False, value=0)

        new_input_ids, new_past_key_values, new_transition_scores = self._shift_padding_to_left(new_input_ids, new_past_key_values, new_transition_scores)
        return new_input_ids, new_past_key_values, new_transition_scores

    def _mask_former_steps(self, input_ids: torch.LongTensor, past_token_lens: torch.LongTensor=None):
        """Mask the former steps"""
        n_paddings = count_left_padding(input_ids, pad_value=self.pad_token_id)
        cur_step_borders_left = n_paddings + past_token_lens

        input_ids = mask_by_borders_2D(input_ids, right_borders=cur_step_borders_left, include_right=False, value=self.pad_token_id)
        return input_ids

    def _step_level_sample_tokens(
        self,
        input_ids: torch.LongTensor, 
        past_key_values: Tuple[Tuple[torch.FloatTensor]],
        num_sampling_sequences: int = 1,
        max_length: int = 2048,
        output_transition_scores: bool = False,
        output_verifier_scores: bool = False,
        output_past_key_values: bool = True,
    ) -> StepSamplingOutput:
        """
        Step-level sampling with calculator (token-level)
        """
        
        if (
            self.pad_token_id is not None
            and len(input_ids.shape) == 2
            and torch.sum(input_ids[:, -1] == self.pad_token_id) > 0
        ):
            print(
                "A decoder-only architecture is being used, but right-padding was detected! For correct "
                "generation results, please set `padding_side='left'` when initializing the tokenizer."
            )
        
        input_ids = input_ids.repeat_interleave(num_sampling_sequences, dim=0)
        if past_key_values is not None:
            past_key_values = tuple(
                tuple(
                    past_key_value.repeat_interleave(num_sampling_sequences, dim=0)
                    for past_key_value in layer_past_key_values
                )
                for layer_past_key_values in past_key_values
            )

        cur_token_lens = count_not_left_padding(input_ids, pad_value=self.pad_token_id)
        stopping_criteria_step = StepStoppingCriteria(cur_token_lens=cur_token_lens, end_token_ids=self.step_end_token_ids.tolist(), pad_token_id=self.pad_token_id, device=self.device)

        outputs = self._sample_tokens_with_calculator(
            input_ids=input_ids,
            past_key_values=past_key_values,
            max_length=max_length,
            stopping_criteria_end=stopping_criteria_step,
            output_transition_scores=output_transition_scores,
            output_past_key_values=output_past_key_values,
        )
        sequences, past_key_values, transition_scores = outputs.sequences, outputs.past_key_values, outputs.transition_scores

        # For the convenience of subsequent operations/processing
        sequences, past_key_values, transition_scores = self._cut_latter_steps(sequences, past_key_values, transition_scores, past_token_lens=cur_token_lens)
        steps = self._mask_former_steps(sequences, past_token_lens=cur_token_lens)

        return StepSamplingOutput(
            sequences=sequences,
            steps=steps,
            transition_scores=outputs.transition_scores,
            verifier_scores=self.verifier_scoring(sequences) if output_verifier_scores else None,
            past_key_values=past_key_values if output_past_key_values else None,
        )

    def _group_step_level_sample(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Tuple[Tuple[torch.FloatTensor]],
        batch_size: int = 1,
        vs_batch_size: int = 1,
        num_sampling_sequences: int = 1,
        max_length: int = 2048,
        output_transition_scores: bool = False,
        output_verifier_scores: bool = False,
        output_past_key_values: bool = True,
    ) -> StepSamplingOutput:

        input_ids = input_ids.repeat_interleave(num_sampling_sequences, dim=0)                       # [n_beam * n_sampling_steps_per_beam, seq_len]
        if past_key_values is not None:
            past_key_values = tuple(tuple(past_key_value.repeat_interleave(num_sampling_sequences, dim=0) for past_key_value in layer_past_key_values) for layer_past_key_values in past_key_values)
        nseqs = input_ids.shape[0]
        n_split = math.ceil(nseqs / batch_size)

        batch_outputs = []
        for i in range(n_split):
            cur_input = input_ids[i*batch_size: min((i+1)*batch_size, nseqs)]
            cur_past_key_values = tuple(tuple(past_key_value[i*batch_size: min((i+1)*batch_size, nseqs)] for past_key_value in layer_past_key_values) for layer_past_key_values in past_key_values) if past_key_values is not None else None

            step_outputs = self._step_level_sample_tokens(                                           # [cur_batch_size, seq_len]
                input_ids=cur_input,
                past_key_values=cur_past_key_values,
                max_length=max_length, 
                output_transition_scores=output_transition_scores,
                output_verifier_scores=False,
                output_past_key_values=output_past_key_values,
            )
            batch_outputs.append(step_outputs)
        
        batch_outputs = self._concat_group_steps(batch_outputs, dim=0)                               # [n_beam * n_sampling_steps_per_beam, seq_len]

        if output_verifier_scores:
            batch_outputs.verifier_scores = self.verifier_scoring(batch_outputs.sequences, batch_size=vs_batch_size)

        return batch_outputs


    def _step_level_majority_tokens(
        self,
        step_ids: torch.FloatTensor, 
        step_generation_scores: torch.FloatTensor, 
    ) -> torch.LongTensor:
        """
        Majority voting at the step level (token-level)

        Parameters:
            step_ids (`torch.FloatTensor`):
                of shape `(num_sampling_sequences, generated_length)`
            step_generation_scores (`torch.FloatTensor`)
        Return:
            index (`torch.LongTensor`):
                of shape `(1,)`
        """

        bsz, seq_len = step_ids.shape

        # identify valid step and only keep the `=...>>` area in the valid step
        cal_result_ids = step_ids
        right_borders = find_rightmost_tokens_positions(cal_result_ids, self.right_mark_token_ids, wnby=False)
        cal_result_ids = mask_by_borders_2D(cal_result_ids, left_borders=right_borders, include_left=False, value=self.pad_token_id)
        left_borders = find_rightmost_tokens_positions(cal_result_ids, self.equal_token_ids, wnby=False)
        cal_result_ids = mask_by_borders_2D(cal_result_ids, right_borders=left_borders, include_right=False, value=self.pad_token_id)
        hit_res = cal_result_ids.ne(self.pad_token_id).any(1)
        
        if hit_res.any():
            # hash to count: first unify the padding loc
            cal_result_ids = shift_padding_to_left_2D(cal_result_ids, pad_value=self.pad_token_id)
            multipliers = torch.pow(torch.full((seq_len,), 31, dtype=cal_result_ids.dtype, device=self.device), torch.arange(seq_len, device=self.device))
            hashes = (cal_result_ids * multipliers).sum(dim=1)
            _, inverse_indices = torch.unique(hashes, return_inverse=True)
            counts = torch.bincount(inverse_indices)
            counts_sorted, indices = torch.sort(counts, descending=True)
            # derive the majority steps and the corresponding count
            if cal_result_ids[inverse_indices == indices[0]].ne(self.pad_token_id).any():  # valid steps are majority
                count_step = counts_sorted[0]
                majority_step_indices = (inverse_indices == indices[0]).nonzero().view(-1)
            elif indices.shape[0] > 1:                                                     # non-steps are majority, consider the second
                count_step = counts_sorted[1]
                majority_step_indices = (inverse_indices == indices[1]).nonzero().view(-1)
        else:                                                                              # there are not valid steps
            count_step = 0
            majority_step_indices = None
        
        # find answers (eos)
        hit_ans = step_ids.eq(self.eos_token_id).any(1)
        ans_indices = torch.where(hit_ans)[0].unique()
        count_ans = ans_indices.shape[0]
        
        # others
        others_indices = torch.where(torch.logical_and(~hit_res, ~hit_ans))[0]

        if count_step > count_ans:
            step_scores = step_generation_scores.index_select(0, majority_step_indices)
            index = majority_step_indices[step_scores.argmax(0)]
        elif count_ans != 0:
            ans_scores = step_generation_scores.index_select(0, ans_indices)
            index = ans_indices[ans_scores.argmax(0)]
        else:
            others_scores = step_generation_scores.index_select(0, others_indices)
            index = others_indices[others_scores.argmax(0)]

        return index

    def verifier_scoring(self, sequences: torch.LongTensor, batch_size: int = 1):
        nseq = sequences.shape[0]
        n_split = math.ceil(nseq / batch_size)

        outputs = []
        for i in range(n_split):
            batch = sequences[i*batch_size: min((i+1)*batch_size, nseq)]
            vscores = self.accelerator.unwrap_model(self.verifier).scoring_sequences(batch)

            outputs.append(vscores)
        return torch.cat(outputs, dim=0) # [bsz,]
    
    def sample_by_steps(
        self,
        qn_str: str = None,
        batch_size: int = 1,
        vs_batch_size: int = 1,
        n_beam: int = 1,
        n_sampling_steps: int = 2,
        max_n_step: int = 10,
        max_step_length: int = 100,
        inference_mode: str = 'beam',
        dedup_mode: int = 0,
    ) -> str:
        """
        Sampling with step-level techniques

        Only support one string by now

        Parameters:
            qn_str (`str`)
            batch_size (`int`):
                used for sampling at each time
            vs_batch_size (`int`):
                batch size of verifier scoring
            n_beam (`int`)
            n_sampling_steps (`int`):
                number of total sampling sequences as next step candidates
            max_n_step (`int`):
                maximum number of steps
            max_step_length (`int`):
                maximum length for a single step
            inference_mode (`str`):
                'verifier', 'majority', or 'beam'
            dedup_mode (`int`):
                0/1
        """
        assert inference_mode in ('majority', 'beam')
        input_ids = self._convert_into_tensors(qn_str)

        if self.max_new_tokens > 0:
            max_length = input_ids.shape[-1] + self.max_new_tokens
        else:
            max_length = self.max_length
        

        if inference_mode == 'majority':
            sequence, all_steps, all_choices, all_step_transition_scores = self._steps_majority(
                input_ids, 
                batch_size=batch_size, 
                n_sampling_steps=n_sampling_steps, 
                max_step_length=max_step_length, 
                max_length=max_length,
            )
            all_scores = all_step_transition_scores

        elif inference_mode == 'beam':
            sequence, all_sequences, all_choices, all_vscores = self._steps_beam_search(
                input_ids, 
                batch_size=batch_size, 
                vs_batch_size=vs_batch_size,
                n_beam=n_beam, 
                n_sampling_steps=n_sampling_steps, 
                max_n_step=max_n_step, 
                max_step_length=max_step_length, 
                max_length=max_length, 
                dedup_mode=dedup_mode,
            )
            all_scores = all_vscores


        completion = self.tokenizer.batch_decode(sequence, skip_special_tokens=True)[0]
        response = completion[len(qn_str):].strip()

        if inference_mode in ('majority', 'verifier'):
            intermediates = [
                {
                    'steps':[
                        {
                            'sample_id': i,
                            'str': self.tokenizer.decode(step, skip_special_tokens=True),
                            'tscore': score.item(),
                        } 
                        if inference_mode == 'majority' else

                        {
                            'sample_id': i,
                            'str': self.tokenizer.decode(step, skip_special_tokens=True),
                            'vscore': score.item(),
                        }
                        for i, (step, score) in enumerate(zip(steps, scores))
                    ],
                    'choice': choice.item(),
                }
                for steps, choice, scores in zip(all_steps, all_choices, all_scores)
            ]
        elif inference_mode == 'beam':
            intermediates = [
                {
                    'sequences':[
                        {
                            'sample_id': i,
                            'str': self.tokenizer.decode(seq, skip_special_tokens=True)[len(qn_str):],
                            'vscore': score.item(),
                        } 
                        for i, (seq, score) in enumerate(zip(sequences, scores))
                    ],
                    'choices': choices.tolist(),
                }
                for sequences, choices, scores in zip(all_sequences, all_choices, all_scores)
            ]
            
        return response, intermediates

    def _steps_majority(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1,
        n_sampling_steps: int = 2,
        max_step_length: int = 100,
        max_length: int = 2048,
    ):
        """
        Majority-voting

        Parameters:
            input_ids (`torch.LongTensor`)
            batch_size (`int`):
                used for sampling at each time
            n_sampling_steps (`int`):
                number of total sampling sequences as next step candidates
            max_step_length (`int`):
                maximal length for a single step
            max_length (`int`):
                maximal length for the complete response
        """

        cur_length = input_ids.shape[-1]

        past_key_values = None
        all_steps = []
        all_step_transition_scores = []
        all_choices = []
        while cur_length < max_length:
            cur_step_max_length = cur_length + max_step_length

            batch_candidates = self._group_step_level_sample(
                input_ids=input_ids,
                past_key_values=past_key_values,
                batch_size=batch_size,
                num_sampling_sequences=n_sampling_steps,
                max_length=min(cur_step_max_length, max_length), 
                output_transition_scores=True,
                output_verifier_scores=False,
            )
            batch_steps = batch_candidates.steps
            batch_transition_scores = batch_candidates.transition_scores

            # select the best steps/sequences
            batch_step_generation_scores = self._cal_generation_scores(batch_transition_scores)
            index = self._step_level_majority_tokens(batch_steps, batch_step_generation_scores)

            sequence = batch_candidates.sequences.index_select(0, index)
            past_key_values = batch_candidates.past_key_values
            if past_key_values is not None:
                past_key_values = tuple(
                    tuple(
                        past_key_value.index_select(0, index)
                        for past_key_value in layer_past_key_values
                    )
                    for layer_past_key_values in past_key_values
                )

            all_steps.append(batch_steps)
            all_step_transition_scores.append(batch_step_generation_scores)
            all_choices.append(index)
            if sequence.eq(self.eos_token_id).any():
                break
            input_ids, past_key_values, _ = self._truncate_left_padding(sequence, past_key_values)
            cur_length = input_ids.shape[-1]

        return sequence, all_steps, all_choices, all_step_transition_scores

    def _steps_beam_search(
        self,
        input_ids: torch.LongTensor,
        batch_size: int = 1,
        vs_batch_size: int = 1,
        n_beam: int = 2,
        n_sampling_steps: int = 2,
        max_n_step: int = 10,
        max_step_length: int = 100,
        max_length: int = 2048,
        dedup_mode: int = 0,
    ):
        """
        Beam search

        Parameters:
            input_ids (`torch.LongTensor`)
            batch_size (`int`):
                used for sampling at each time
            vs_batch_size (`int`):
                batch size of verifier scoring
            n_beam (`int`):
                number of kept sequences when progressing
            n_sampling_steps (`int`):
                number of total sampling sequences as next step candidates
            max_n_step (`int`)
            max_step_length (`int`):
                maximal length for a single step
            max_length (`int`):
                maximal length for the complete response
            dedup_mode (`int`):
                linguistics-level (mode=1); 0 indicates "no"
        """

        assert self.verifier is not None

        assert n_sampling_steps % n_beam == 0
        n_sampling_steps_per_beam = n_sampling_steps // n_beam

        input_ids = input_ids.repeat_interleave(n_beam, dim=0) # [n_beam, seq_len]

        cur_length = input_ids.shape[-1]

        past_key_values = None
        all_sequences = []
        all_vscores = []
        all_choices = []
        cur_step = 0
        while cur_length < max_length and cur_step < max_n_step:
            cur_step_max_length = cur_length + max_step_length

            batch_candidates = self._group_step_level_sample(
                input_ids=input_ids,
                past_key_values=past_key_values,
                batch_size=batch_size,
                vs_batch_size=vs_batch_size,
                num_sampling_sequences=n_sampling_steps_per_beam,
                max_length=min(cur_step_max_length, max_length), 
                output_transition_scores=False,
                output_verifier_scores=True,
            )
            batch_sequences = batch_candidates.sequences         # [n_beam * n_sampling_steps_per_beam, seq_len]
            batch_vscores = batch_candidates.verifier_scores

            # select the best steps/sequences
            hvscores = self._highlight_unique_sequences(batch_sequences, batch_vscores, dedup_mode=dedup_mode)
            _, indices = torch.topk(hvscores, k=n_beam, dim=0, largest=True)

            sequences = batch_sequences.index_select(0, indices) # [n_beam, seq_len]
            past_key_values = batch_candidates.past_key_values
            if past_key_values is not None:
                past_key_values = tuple(
                    tuple(
                        past_key_value.index_select(0, indices)
                        for past_key_value in layer_past_key_values
                    )
                    for layer_past_key_values in past_key_values
                )
            vscores = batch_vscores.index_select(0, indices)

            all_sequences.append(batch_sequences)
            all_vscores.append(batch_vscores)
            all_choices.append(indices)
            if sequences.eq(self.eos_token_id).any(1).all():
                break
            input_ids, past_key_values, _ = self._truncate_left_padding(sequences, past_key_values)
            cur_length = input_ids.shape[-1]

            cur_step += 1

        # final selection
        _, best_index = torch.topk(vscores, k=1, dim=0, largest=True)
        all_sequences.append(sequences)
        all_vscores.append(vscores)
        all_choices.append(best_index)

        sequence = sequences.index_select(0, best_index)
        return sequence, all_sequences, all_choices, all_vscores

    def _highlight_unique_sequences(self, sequences: torch.LongTensor, verifier_scores: torch.FloatTensor, dedup_mode: int=0) -> torch.FloatTensor:
        """
        Prioritize unique sequences: linguistics-level (mode=1)
        """
        if dedup_mode == 0:
            return verifier_scores
        
        seq_len = sequences.shape[-1]
        
        seqs = shift_padding_to_left_2D(sequences, pad_value=self.pad_token_id)
        multipliers = torch.pow(torch.full((seq_len,), 31, dtype=seqs.dtype, device=self.device), torch.arange(seq_len, device=self.device))
        hashes = (seqs * multipliers).sum(dim=1)

        unique_hashes = torch.unique(hashes)
        hightlighted_indices = (unique_hashes[:, None] == hashes[None, :]).float().argmax(dim=1)

        highlighted_vscores = verifier_scores.clone()
        highlighted_vscores[hightlighted_indices] += 100
        return highlighted_vscores


    def _resize_step_level_outputs_by_beam(self, step_outputs: StepSamplingOutput, n_beam: int, n_sampling_steps: int) -> StepSamplingOutput:

        sequences = step_outputs.sequences.view(n_beam, n_sampling_steps, -1)
        steps = step_outputs.steps.view(n_beam, n_sampling_steps, -1)

        transition_scores = step_outputs.transition_scores
        if transition_scores is not None:
            transition_scores = transition_scores.view(n_beam, n_sampling_steps, -1)

        verifier_scores = step_outputs.verifier_scores
        if verifier_scores is not None:
            verifier_scores = verifier_scores.view(n_beam, n_sampling_steps, -1)

        past_key_values = step_outputs.past_key_values
        if past_key_values is not None:
            past_key_values = tuple(
                tuple(
                    past_key_value.view(n_beam, n_sampling_steps, *past_key_value.shape[1:])
                    for past_key_value in layer_past_key_values
                ) 
                for layer_past_key_values in past_key_values
            )

        return StepSamplingOutput(
            sequences=sequences,
            steps=steps,
            transition_scores=transition_scores,
            verifier_scores=verifier_scores,
            past_key_values=past_key_values,
        )

    def _concat_group_tensors(self, tensor_list: List[torch.Tensor], left_padding = True, pad_value: int = 0, dim: int = 0):
        max_len = max(tensor.shape[-1] for tensor in tensor_list)
        if left_padding:
            tensor_list = [F.pad(tensor, (max_len - tensor.shape[-1], 0), value=pad_value) for tensor in tensor_list]
        else:
            tensor_list = [F.pad(tensor, (0, max_len - tensor.shape[-1]), value=pad_value) for tensor in tensor_list]

        tensors = torch.concat(tensor_list, dim=dim)
        return tensors

    def _concat_group_past_key_values(self, past_key_values: List[Tuple[Tuple[torch.FloatTensor]]], token_padding_lens: torch.LongTensor, dim: int = 0):
        # w/o beam: (bsz, n_heads, cache_len, embed_size)
        # w beam: (n_beam, n_sampling_steps_per_beam, n_heads, cache_len, embed_size)

        cache_lens = torch.LongTensor([cache[0][0].shape[-2] for cache in past_key_values]).to(self.device)
        padded_cache_lens = token_padding_lens + cache_lens
        min_cache_len = padded_cache_lens.min()
        cut_cache_lens = padded_cache_lens - min_cache_len

        past_key_values = tuple(
            tuple(
                torch.cat(
                    [F.pad(tensor.transpose(-2, -1), (token_padding_lens[i], -cut_cache_lens[i]), value=0).transpose(-2, -1) for i, tensor in enumerate(tensor_tuples)], 
                    dim=dim
                )
                for tensor_tuples in zip(*layer_tuples)
            )
            for layer_tuples in zip(*past_key_values)
        )
        return past_key_values

    def _concat_group_steps(self, instances: List[StepSamplingOutput], dim: int = 0):
        sequences, steps, transition_scores, verifier_scores, past_key_values = tuple([instance.get(key) for instance in instances] for key in ("sequences", "steps", "transition_scores", "verifier_scores", "past_key_values"))
        
        seq_lens = torch.LongTensor([seq.shape[-1] for seq in sequences]).to(self.device)
        max_seq_len = seq_lens.max()
        token_padding_lens = max_seq_len - seq_lens

        sequences = self._concat_group_tensors(sequences, pad_value=self.pad_token_id, dim=dim)
        steps = self._concat_group_tensors(steps, pad_value=self.pad_token_id, dim=dim)
        transition_scores = self._concat_group_tensors(transition_scores, pad_value=0, dim=dim) if transition_scores[0] is not None else None
        verifier_scores = torch.cat(verifier_scores, dim=dim) if verifier_scores[0] is not None else None

        past_key_values = self._concat_group_past_key_values(past_key_values, token_padding_lens, dim=dim) if past_key_values[0] is not None else None

        return StepSamplingOutput(
            sequences=sequences,
            steps=steps,
            transition_scores=transition_scores,
            verifier_scores=verifier_scores,
            past_key_values=past_key_values,
        )
    
    def _flatten_step_level_outputs_beam(self, step_outputs: StepSamplingOutput) -> StepSamplingOutput:
        sequences, steps, transition_scores, verifier_scores, past_key_values = step_outputs.sequences, step_outputs.steps, step_outputs.transition_scores, step_outputs.verifier_scores, step_outputs.past_key_values

        sequences = sequences.view(-1, sequences.shape[-1])
        steps = steps.view(-1, steps.shape[-1])
        transition_scores = transition_scores.view(-1, transition_scores.shape[-1]) if transition_scores is not None else None
        verifier_scores = verifier_scores.view(-1,) if verifier_scores is not None else None

        if past_key_values is not None:
            _, _, n_heads, cache_len, embed_size = past_key_values[0][0].shape
            past_key_values = tuple(tuple(past_key_value.view(-1, n_heads, cache_len, embed_size) for past_key_value in layer_past_key_values) for layer_past_key_values in past_key_values)

        return StepSamplingOutput(
            sequences=sequences,
            steps=steps,
            transition_scores=transition_scores,
            verifier_scores=verifier_scores,
            past_key_values=past_key_values,
        )

