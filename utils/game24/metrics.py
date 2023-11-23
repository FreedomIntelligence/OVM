import os
import torch
import numpy as np
from typing import Optional, List, Dict, Set, Any, Union
import torch.distributed as dist
import re
from utils.game24.decoding import extract_expressions, get_answer_label



class GeneratorAnswerAcc:
    def __init__(self, n_data: int):
        self.n_data = n_data

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.corrs = []
        self.gather = False

    @torch.inference_mode(mode=True)
    def __call__(self, completions: List[str], questions: List[str]):
        expressions = extract_expressions(completions)

        corrs = [float(get_answer_label(expression, question)) == True for expression, question in zip(expressions, questions)]

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
        self.questions = []
    
    def start_new_sol_epoch(self):
        self.cur_answers = []
        self.cur_questions = []

    def end_the_sol_epoch(self):
        
        if self.world_size != 1:            
            gathered_answers, gathered_questions = tuple([None] * self.world_size for _ in range(2))
            for obj, container in [
                (self.cur_answers, gathered_answers), 
                (self.cur_questions, gathered_questions), 
            ]:
                dist.all_gather_object(container, obj)

            flatten_answers, flatten_questions = [], []
            for answers_gpus, questions_gpus in zip(zip(*gathered_answers), zip(*gathered_questions)):
                for answers, questions in zip(answers_gpus, questions_gpus):
                    flatten_answers.extend(answers)
                    flatten_questions.extend(questions)

        else:
            flatten_answers, flatten_questions = tuple([item for sublist in container for item in sublist]
                                                    for container in [self.cur_answers, self.cur_questions])

        self.answers.append(flatten_answers[:self.n_data])
        self.questions.append(flatten_questions[:self.n_data])

    @torch.inference_mode(mode=True)
    def __call__(self, completions: List[str], questions: List[str]):
        expressions = extract_expressions(completions)

        self.cur_answers.append(expressions)
        self.cur_questions.append(questions)

    def get_metric(self, n_solution: int=3, reset=True):        
        # [n_question, n_solution]
        answers = self.answers[:n_solution]
        # [n_question]
        questions = self.questions[:n_solution][0]

        pass_k = np.mean([is_passk(expressions, question) for expressions, question in zip(answers, questions)])
        acc_majority = np.mean([is_majority(expressions, question) for expressions, question in zip(answers, questions)])


        if reset:
            self.answers = []
            self.questions = []
        return pass_k, acc_majority


def is_passk(expressions, question):
   return any(get_answer_label(expression, question) for expression in expressions)


def is_majority(expressions, question):
    repres = [get_semantics(expr) for expr in expressions]
    final_repre = max(repres, key=repres.count)
    index = repres.index(final_repre)
    return get_answer_label(expressions[index], question)


def get_semantics(expression):
    numbers = re.findall(r'\d+', expression)
    symbols = re.findall(r'[+\-\*\/]', expression)

    try:
        value = eval(expression)
    except:
        value = None

    value = str(value)
    if value[-2:] == '.0':
        value = value[:-2]
    return tuple(sorted(numbers) + sorted(symbols) + [f'value={value}'])


