import os
import torch
import numpy as np
import torch.distributed as dist
from typing import Optional, List, Dict, Set, Any, Union
from utils.constants import IGNORE_INDEX



class VerifierClassificationAcc:
    def __init__(self, n_data: int):
        self.n_data = n_data

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        
        self.scores = []
        self.gts = []
        self.gather = False

    @torch.inference_mode(mode=True)
    def __call__(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        bsz, n_seq = v_labels.shape
        index = ((n_seq - 1) - v_labels.ne(IGNORE_INDEX).flip(dims=[1]).float().argmax(1)).view(-1, 1)
        
        scores = v_scores.squeeze(-1).gather(1, index).squeeze()
        gts = v_labels.gather(1, index).squeeze()
        
        self.scores.append(scores.tolist())
        self.gts.append(gts.tolist())

    def get_metric(self, thres: float=0.5, reset=True):
        if not self.gather:
            if self.world_size != 1:
                gathered_scores, gathered_gts = tuple([None] * self.world_size for _ in range(2))
                for obj, container in [
                    (self.scores, gathered_scores), 
                    (self.gts, gathered_gts), 
                ]:
                    dist.all_gather_object(container, obj)
                    
                flatten_scores, flatten_gts = [], []
                for scores_gpus, gts_gpus in zip(zip(*gathered_scores), zip(*gathered_gts)):
                    for scores, gts in zip(scores_gpus, gts_gpus):
                        flatten_scores.extend(scores)
                        flatten_gts.extend(gts)

            else:
                flatten_scores, flatten_gts = tuple([item for sublist in container for item in sublist]
                                                    for container in [self.scores, self.gts])

            self.scores = flatten_scores[:self.n_data]
            self.gts = flatten_gts[:self.n_data]
            self.gather = True

        
        pred = (np.array(self.scores) > thres)
        corrs = np.where(np.array(self.gts).astype(bool), pred, ~pred)
        acc = (sum(corrs) / len(corrs))

        if reset:
            self.scores = []
            self.gts = []
            self.gather = False
        return acc


class VerifierMPk:
    def __init__(self, n_data: int, n_solution_per_problem: int):
        self.n_data = n_data
        self.n_solution_per_problem = n_solution_per_problem

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.preds = []
        self.gts = []
        self.gather = False

    @torch.inference_mode(mode=True)
    def __call__(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        bsz, n_seq = v_labels.shape
        index = ((n_seq - 1) - v_labels.ne(IGNORE_INDEX).flip(dims=[1]).float().argmax(1)).view(-1, 1)
        
        preds = v_scores.squeeze(-1).gather(1, index).squeeze()
        gts = v_labels.gather(1, index).squeeze()
        
        self.preds.append(preds.tolist())
        self.gts.append(gts.tolist())

    def get_metric(self, k, reset=True):
        if not self.gather:
            if self.world_size != 1:
                gathered_preds, gathered_gts = tuple([None] * self.world_size for _ in range(2))
                for obj, container in [
                    (self.preds, gathered_preds), 
                    (self.gts, gathered_gts), 
                ]:
                    dist.all_gather_object(container, obj)
                    
                flatten_preds, flatten_gts = [], []
                for preds_gpus, gts_gpus in zip(zip(*gathered_preds), zip(*gathered_gts)):
                    for preds, gts in zip(preds_gpus, gts_gpus):
                        flatten_preds.extend(preds)
                        flatten_gts.extend(gts)

            else:
                flatten_preds, flatten_gts = tuple([item for sublist in container for item in sublist]
                                                    for container in [self.preds, self.gts])

            self.preds = flatten_preds[:self.n_data]
            self.gts = flatten_gts[:self.n_data]
            self.gather = True

        preds = np.array(self.preds).reshape(-1, self.n_solution_per_problem)
        gts = np.array(self.gts).reshape(-1, self.n_solution_per_problem)

        indices = np.argsort(-preds, axis=1)
        gts = np.take_along_axis(gts, indices, axis=1)

        # [n_question, k]
        gts_topk = gts[:, :k]

        # how portion of solutions predicted topest are really correct
        mpk = gts_topk.mean(1).mean(0)

        if reset:
            self.preds = []
            self.gts = []
            self.gather = False
        return mpk
    

class GenWithVerifierAcc:
    def __init__(self, n_data: int, n_solution_per_problem: int):
        self.n_data = n_data
        self.n_solution_per_problem = n_solution_per_problem

        self.world_size = dist.get_world_size() if dist.is_initialized() else 1

        self.preds = []
        self.gts = []
        self.gather = False

    @torch.inference_mode(mode=True)
    def __call__(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        bsz, n_seq = v_labels.shape
        index = ((n_seq - 1) - v_labels.ne(IGNORE_INDEX).flip(dims=[1]).float().argmax(1)).view(-1, 1)
        
        preds = v_scores.squeeze(-1).gather(1, index).squeeze()
        gts = v_labels.gather(1, index).squeeze()
        
        self.preds.append(preds.tolist())
        self.gts.append(gts.tolist())
    

    def get_metric(self, k, reset=True):
        if not self.gather:
            if self.world_size != 1:
                gathered_preds, gathered_gts = tuple([None] * self.world_size for _ in range(2))
                for obj, container in [
                    (self.preds, gathered_preds), 
                    (self.gts, gathered_gts), 
                ]:
                    dist.all_gather_object(container, obj)
                    
                flatten_preds, flatten_gts = [], []
                for preds_gpus, gts_gpus in zip(zip(*gathered_preds), zip(*gathered_gts)):
                    for preds, gts in zip(preds_gpus, gts_gpus):
                        flatten_preds.extend(preds)
                        flatten_gts.extend(gts)

            else:
                flatten_preds, flatten_gts = tuple([item for sublist in container for item in sublist]
                                                    for container in [self.preds, self.gts])

            self.preds = flatten_preds[:self.n_data]
            self.gts = flatten_gts[:self.n_data]
            self.gather = True


        preds = np.array(self.preds).reshape(-1, self.n_solution_per_problem)
        gts = np.array(self.gts).reshape(-1, self.n_solution_per_problem)
        gts = gts[:, :k]
        preds = preds[:, :k]

        indices = np.argsort(-preds, axis=1)
        gts = np.take_along_axis(gts, indices, axis=1)
        
        acc = gts[:, 0].mean(0)

        if reset:
            self.preds = []
            self.gts = []
            self.gather = False
        return acc
    
