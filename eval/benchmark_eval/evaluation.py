import numpy as np
import os
from torch.utils.data import DataLoader
import torch
from typing import Dict, Tuple, Sequence, List, Union, Any
import json
import re
from tqdm import tqdm
import gc
from accelerate import Accelerator
from utils import Agent
from benchmark_eval.data_utils import *




class EvaluationBase:
    """
    The evaluation pipeline:
        load datasets (from benchmark classes)
        model generation and decoding
        metrics calculation
        save output files and metrics file

    Subclass should config:
        benchmark (`BenchmarkBase`)

    Subclass should implement:
        _extract_answer(self, response: str):
            decode the answer, especially for the objective questions

        _compute_metrics(self, samples: List[dict], task_name: str):
            specify what metrics to compute for different subtasks

        _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]):
            specify whether and how to aggregate metrics over different subtasks
    """
    # output: model_results_dir - benchmark_eval/results/$benchmark_name/$model_id/$setting
    results_dir = 'benchmark_eval/results'
    def __init__(
        self, 
        agent: Agent, 
        benchmark: BenchmarkBase,
        batch_size: int = 1,
        setting: str = 'zero_shot',
        n_shot: int = 5,
        accelerator: Accelerator = None,
    ):
        self.agent = agent
        self.benchmark = benchmark
        self.batch_size = batch_size
        self.setting = setting
        self.n_shot = n_shot
        self.accelerator = accelerator
        assert setting in ('zero_shot', 'few_shot'), (f"setting should be in ('zero_shot', 'few_shot'), but it is {setting}")


        self.benchmark_name = benchmark.benchmark_name
        self.subtasks = benchmark.subtasks
        self.model_id = self.agent.model_id
        self.collate_fn = self.agent.collate_fn
        self.stage = self.agent.stage

        self.benchmark_results_dir = os.path.join(self.results_dir, self.benchmark_name)
        self.model_results_dir = os.path.join(self.benchmark_results_dir, self.model_id)
        self.setting_results_dir = os.path.join(self.model_results_dir, self.setting)
        # self._create_folder(self.setting_results_dir)
        # self._create_folder(self.benchmark_results_dir)
        # self._create_folder(self.model_results_dir)
    
    @staticmethod
    def _create_folder(path: str):
        """Create a folder for the path if there isn't"""
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def compute_acc(responses_answer: list, answers: list) -> float:
        """Compute accuracy with extracted responses and ground answers"""
        assert isinstance(responses_answer, list) and isinstance(answers, list)
        assert len(responses_answer) == len(answers)
        return np.mean([r == a for r, a in zip(responses_answer, answers)])

    def _extract_answer(self, response: str, task_name: str) -> str:
        """derive the required answer"""
        raise NotImplementedError

    def _extract_answers(self, responses: List[str], task_name: str) -> List[str]:
        """extract all the required answers"""
        assert isinstance(responses, list) and isinstance(responses[0], str)
        return [self._extract_answer(response, task_name) for response in responses]
    
    def _zip_qra(self, dataset: List[dict], responses: List[str], responses_answer: List[str]) -> List[dict]:
        """Pack the response and its decoded answer of each sample with the corresponding query and answer"""
        assert len(dataset) == len(responses) == len(responses_answer), f'dataset, responses and responses_answer must have the same length, but are {len(dataset)}, {len(responses)} and {len(responses_answer)}'
        samples = []
        for data, response, response_answer in zip(dataset, responses, responses_answer):
            samples.append({
                **data,
                'response': response,
                'response_answer': response_answer,
            })
        return samples

    def _compute_metrics(self, samples: List[dict], task_name: str):
        """
        Compute all metrics for samples. You can compute different metrics for different tasks
        Args:
            samples (`List[dict]`):
                List of samples
            task_name (`str`)

        Return:
            metrics (`Dict[str, float]`):
                {
                    'metric_name': metric
                }
        """
        raise NotImplementedError

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics for subtasks
        Args:
            metrics (`Dict[str, Dict[str, float]]`): 
                {'task_name': {'metric_name': metric}}
            lens (`Dict[str, int]`): 
                {'task_name': #task_examples}
        """
        raise NotImplementedError
        
    def save_response(self, samples: List[dict], task_name: str):
        """Save responses to `$results/$benchmark_name/$model_id/file`"""
        assert isinstance(task_name, str) and (isinstance(samples, list) and isinstance(samples[0], dict))

        save_file = os.path.join(self.setting_results_dir, f'{task_name}.jsonl')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w', encoding='utf-8') as f:
            f.writelines([json.dumps(sample, ensure_ascii=False) + '\n'  for sample in samples])
        print(f'Save responses to {save_file}')

    def save_aggr_metrics(self, aggr_metrics: dict):
        """Save aggr_metrics to `$results/$benchmark_name/$model_id/metrics.json`"""
        assert isinstance(aggr_metrics, dict)

        save_file = os.path.join(self.setting_results_dir, 'metrics.json')
        os.makedirs(os.path.dirname(save_file), exist_ok=True)
        with open(save_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(aggr_metrics, indent=4, ensure_ascii=False) + '\n')
        print(f'Save metrics to {save_file}')

    def _count_examples(self, datasets: Dict[str, List[dict]]) -> Dict[str, int]:
        return {
            task_name: len(datasets[task_name]) 
            for task_name in datasets.keys()
        }

    def _eval_task(
        self, 
        task_name: str, 
        dataset: List[dict],
        save_responses: bool = True,
    ) -> Dict[str, float]:
        """
        Evaluate and return the metrics for one dataset
        Args:
            task_name (`str`)
            dataset (`List[dict]`)
        Return:
            metrics (`Dict[str, float]`):
                {
                    'metric_name': metric
                }
        """
        assert isinstance(task_name, str) and isinstance(dataset, list)

        response_list = self.agent.generate_from_dataset(
            query_list=[data['prompted_query'] for data in dataset], 
            response_with=dataset[0]['response_with'],
            batch_size=self.batch_size, 
            description=task_name,
        )
        response_answer_list = self._extract_answers(response_list, task_name)

        samples = self._zip_qra(dataset, response_list, response_answer_list)
        if save_responses and self.accelerator.is_main_process:
            self.save_response(samples, task_name)

        metrics = self._compute_metrics(samples, task_name)
        self.accelerator.print(metrics)

        gc.collect(); torch.cuda.empty_cache()
        return metrics

    def _eval_datasets(
        self,
        all_datasets: Dict[str, List[dict]],
        save_responses: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate and return the metrics for all datasets
        Return:
            metrics (`Dict[str, dict]`):
                {
                    'task_name': {
                            'metric_name': metric
                        }
                }
        """
        return {
            task_name: self._eval_task(task_name, all_datasets[task_name], save_responses=save_responses) 
            for task_name in all_datasets.keys()
        }

    def evaluate(self):
        all_datasets = self.benchmark.get_datasets(self.setting, self.n_shot, self.stage)
        lens = self._count_examples(all_datasets)
        metrics = self._eval_datasets(all_datasets, save_responses=True)
        aggr_metrics = self._aggregate_metrics(metrics, lens)
        if self.accelerator.is_main_process:
            self.save_aggr_metrics(aggr_metrics)






class MMLUArabicEvaluation(EvaluationBase):
    # output: model_results_dir - benchmark_eval/results/$setting/MMCU/$model_id
    
    def __init__(self, 
                 agent: Agent, 
                 batch_size: int,
                 setting: str='zero_shot',
                 n_shot: int=5,
                 accelerator: Accelerator=None):
        
        super(MMLUArabicEvaluation, self).__init__(agent=agent, 
                                             batch_size=batch_size,
                                             setting=setting,
                                             n_shot=n_shot,
                                             benchmark=MMLUArbicBenchmark(),
                                             accelerator=accelerator)
        
        self.top_categories = self.benchmark.top_categories
        self.categories = self.benchmark.categories

    def _extract_answer(self, response: str, task_name: str) -> str:
        """derive the required answer"""
        def standardize_options(options):
            return ''.join(sorted(set(options)))
        return standardize_options(re.findall(r'[ABCD]', response.split('\n')[0]))
    
    def _compute_metrics(self, samples: List[dict], task_name: str):
        responses_answer = [sample['response_answer'] for sample in samples]
        answers = [sample['answer'] for sample in samples]
        return {
            'Accuracy': self.compute_acc(responses_answer, answers)
        }

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate 'Humanities', 'Social Science', 'STEM', 'Other'
        """
        aggr_metrics = {}
        metric_names = list(metrics.values())[0].keys()

        def aggregate(top_subject, top_category):
            aggr_metrics[top_category][top_subject] = {}
            subjects = self.categories[top_subject]
            for subject in subjects:
                aggr_metrics[top_category][top_subject][subject] = metrics[subject]

            aggr_metrics[top_category][top_subject]['average'] = {}
            aggr_metrics[top_category][top_subject]['overall'] = {}
            for metric_name in metric_names:
                aggr_metrics[top_category][top_subject]['average'][metric_name] = np.mean([metrics[subject][metric_name] for subject in subjects])
                aggr_metrics[top_category][top_subject]['overall'][metric_name] = np.sum([lens[subject] * metrics[subject][metric_name] for subject in subjects]) / np.sum([lens[subject] for subject in subjects])
        
        for top_category in self.top_categories:
            aggr_metrics[top_category] = {}
            for subject in self.top_categories[top_category]:
                aggregate(subject, top_category)
                
        return aggr_metrics




class EXAMSEvaluation(EvaluationBase):
    # output: model_results_dir - benchmark_eval/results/$setting/MMCU/$model_id
    
    def __init__(self, 
                 agent: Agent, 
                 batch_size: int,
                 setting: str='zero_shot',
                 n_shot: int=5,
                 accelerator: Accelerator=None):
        
        super(EXAMSEvaluation, self).__init__(agent=agent, 
                                             batch_size=batch_size,
                                             setting=setting,
                                             n_shot=n_shot,
                                             benchmark=EXAMSBenchmark(),
                                             accelerator=accelerator)

    def _extract_answer(self, response: str, task_name: str) -> str:
        """derive the required answer"""
        def standardize_options(options):
            return ''.join(sorted(set(options)))
        return standardize_options(re.findall(r'[ABCD]', response.split('\n')[0]))
    
    def _compute_metrics(self, samples: List[dict], task_name: str):
        responses_answer = [sample['response_answer'] for sample in samples]
        answers = [sample['answer'] for sample in samples]
        return {
            'Accuracy': self.compute_acc(responses_answer, answers)
        }

    def _aggregate_metrics(self, metrics: Dict[str, Dict[str, float]], lens: Dict[str, int]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate medicine and eduction
        """
        aggr_metrics = {}
        metric_names = list(metrics.values())[0].keys()

        def aggregate():
            for topic in self.subtasks:
                aggr_metrics[topic] = metrics[topic]

            aggr_metrics['average'] = {}
            aggr_metrics['overall'] = {}
            for metric_name in metric_names:
                aggr_metrics['average'][metric_name] = np.mean([metrics[topic][metric_name] for topic in self.subtasks])
                aggr_metrics['overall'][metric_name] = np.sum([lens[topic] * metrics[topic][metric_name] for topic in self.subtasks]) / np.sum([lens[topic] for topic in self.subtasks])

        aggregate()
        return aggr_metrics
    



