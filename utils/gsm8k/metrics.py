import os
import torch
import numpy as np
from typing import Optional, List, Dict, Set, Any, Union
import torch.distributed as dist
from utils.gsm8k.decoding import INVALID_ANS, extract_answers, get_answer_label



class GeneratorAnswerAcc:
    def __init__(self, n_data: int):
        self.n_data = n_data

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.corrs = []
        self.gather = False

    @torch.inference_mode(mode=True)
    def __call__(self, completions: List[str], gts: List[str]):
        answers = extract_answers(completions)

        corrs = [float(get_answer_label(answer, gt) == True) for answer, gt in zip(answers, gts)]

        self.corrs.append(corrs)

    def get_metric(self, reset=True):
        if not self.gather:
            if self.world_size != 1:
                gathered_corrs = [None] * self.world_size
                for obj, container in [
                    (self.corrs, gathered_corrs), 
                ]:
                    dist.all_gather_object(container, obj)
                    
                flatten_corrs = []
                for corrs_gpus in zip(*gathered_corrs):
                    for corrs in corrs_gpus:
                        flatten_corrs.extend(corrs)

            else:
                flatten_corrs = [item for sublist in self.corrs for item in sublist]

            self.corrs = flatten_corrs[:self.n_data]
            self.gather = True

        acc = (sum(self.corrs) / len(self.corrs))

        if reset:
            self.corrs = []
            self.gather = False
        return acc


class MultiSamplingAnswerAcc:
    def __init__(self, n_data: int = None):
        self.n_data = n_data

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.answers = []
        self.gts = []
    
    def start_new_sol_epoch(self):
        self.cur_answers = []
        self.cur_gts = []

    def end_the_sol_epoch(self):
        
        if self.world_size != 1:            
            gathered_answers, gathered_gts = tuple([None] * self.world_size for _ in range(2))
            for obj, container in [
                (self.cur_answers, gathered_answers), 
                (self.cur_gts, gathered_gts), 
            ]:
                dist.all_gather_object(container, obj)

            flatten_answers, flatten_gts = [], []
            for answers_gpus, gts_gpus in zip(zip(*gathered_answers), zip(*gathered_gts)):
                for answers, gts in zip(answers_gpus, gts_gpus):
                    flatten_answers.extend(answers)
                    flatten_gts.extend(gts)

        else:
            flatten_answers, flatten_gts = tuple([item for sublist in container for item in sublist]
                                                    for container in [self.cur_answers, self.cur_gts])

        self.answers.append(flatten_answers[:self.n_data])
        self.gts.append(flatten_gts[:self.n_data])


    @torch.inference_mode(mode=True)
    def __call__(self, completions: List[str], gts: List[str]):
        answers = extract_answers(completions)
        
        answers = [float(a) if a != INVALID_ANS else float('nan') for a in answers]
        gts = [float(gt) for gt in gts]

        self.cur_answers.append(answers)
        self.cur_gts.append(gts)
        

    def get_metric(self, n_solution: int=3, reset=True):

        assert all(x == self.gts[0] for x in self.gts)

        # [n_question]
        gts = np.array(self.gts[0])
        # [n_question, n_solution]
        answers = np.stack(self.answers[:n_solution], axis=1)
        # print('answers:', answers.shape)

        pass_k = (answers == gts.reshape((-1, 1))).any(1).mean(0)
        acc_majority = np.mean([is_majority(a, gt, ignore=float('nan')) for a, gt in zip(answers, gts)])

        if reset:
            self.gts = []
            self.answers = []
        return pass_k, acc_majority
    


def is_passk(answers, gt):
   return gt in answers

def is_majority(answers, gt, ignore = INVALID_ANS):
   filter_answers = list(filter(lambda x: x!=ignore, answers))
   final_answer = max(filter_answers, key=filter_answers.count)
   return final_answer == gt


