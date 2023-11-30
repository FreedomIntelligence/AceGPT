import re
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
import transformers
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, Sequence, List, Union, Any
import torch.nn.functional as F
import os
import torch
import gc
import json
import logging
import multiprocessing
from torch.utils.data import DataLoader
from accelerate import Accelerator
import torch.distributed as dist
from utils.configs import load_gconfig, load_model_and_tokenizer, load_system_prompt, load_stage


def get_special_encoding():  # lookup the specified method to load model and tokenizer. it can be indexed by either model_id (prior) or model_name
    return {
        'acegpt': encoding_acegpt,
    }


class Agent:
    def __init__(
        self, 
        model_id: str = None,
        model_name: str = None,
        model: PreTrainedModel = None, 
        tokenizer: Union[PreTrainedTokenizerFast, PreTrainedTokenizer] = None, 
        stage: int = None,
        system_prompt: str = None,
        gconfig: GenerationConfig = None,
        accelerator: Accelerator = None,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.stage = stage
        self.system_prompt = system_prompt
        self.gconfig = gconfig
        self.accelerator = accelerator

        self.device = self.accelerator.device
        self.pad_token_id = self.tokenizer.pad_token_id
        self.collate_fn = transformers.DataCollatorForSeq2Seq(self.tokenizer, return_tensors="pt", padding=True)

        self.model = self.accelerator.prepare_model(self.model)
        self.model.eval()

    @classmethod
    def from_model_id(
        cls, 
        model_id: str, 
        generation_type: str = 'greedy',
        accelerator: Accelerator = None,
    ):
        model_name, model, tokenizer = load_model_and_tokenizer(model_id)
        stage = load_stage(model_id)
        system_prompt = load_system_prompt(model_id)
        gconfig = load_gconfig(generation_type=generation_type)
        return cls(
            model_id=model_id,
            model_name=model_name,
            model=model,
            tokenizer=tokenizer,
            stage=stage,
            system_prompt=system_prompt,
            gconfig=gconfig,
            accelerator=accelerator,
        )

    def decode(self, token_ids_list: List[List[int]]):
        return self.tokenizer.batch_decode(token_ids_list, skip_special_tokens=True) 

    def encode_sequences(self, sequences: List[str], response_with: str='') -> List[List[int]]:
        assert isinstance(sequences, list) and isinstance(sequences[0], str)

        special_encoding = get_special_encoding()
        encoding_fn = special_encoding.get(self.model_id, special_encoding.get(self.model_name, None))
        if encoding_fn is None:
            return [
                {'input_ids': self.tokenizer.encode(self.system_prompt.format(question=seq) + response_with, add_special_tokens=True)}
                for seq in sequences
            ]
        else:
            return [
                {'input_ids': encoding_fn(self.system_prompt.format(question=seq) + response_with, tokenizer=self.tokenizer)}
                for seq in sequences
            ]
    
    @torch.inference_mode(mode=True)
    def generate(
        self, 
        input_ids: torch.LongTensor, 
        attention_mask: torch.FloatTensor = None,
        **kwargs,
    ):
        n_seq = input_ids.shape[-1]

        output_ids = self.accelerator.unwrap_model(self.model).generate(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            generation_config=self.gconfig,
            **kwargs,
        )[..., n_seq:]
        return output_ids

    def generate_from_dataset(
        self, 
        query_list: List[str], 
        response_with: str = '',
        batch_size: int = 1,
        description: str = None,
        device_placement: bool = True,
        **kwargs,
    ) -> List[str]:
        inputs = self.encode_sequences(query_list, response_with=response_with)
        dataloader = DataLoader(inputs, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=not device_placement, collate_fn=self.collate_fn)
        dataloader = self.accelerator.prepare_data_loader(dataloader, device_placement=device_placement)

        output_ids_list = []
        dataloader_iterator = tqdm(dataloader, desc=description) if self.accelerator.is_main_process else dataloader
        for batch in dataloader_iterator:
            output_ids: torch.LongTensor = self.generate(**batch.to(self.device), **kwargs)
            if self.accelerator.num_processes != 1:
                all_output_ids = [None] * dist.get_world_size()
                dist.all_gather_object(all_output_ids, output_ids.tolist())
                all_output_ids = [item for sublist in all_output_ids for item in sublist]
            else:
                all_output_ids = output_ids.tolist()

            output_ids_list.extend(all_output_ids)

        responses = self.decode(output_ids_list[:len(query_list)])
        return responses

    def response_str2str(self, query_list: List[str], setting: str='zero_shot') -> List[str]:
        """
        Sample one response for each query
        Args:
            query_list (`List[str]`)
                List of queries
            setting (`str`)
        Return:
            responses (`List[str]`)
                List of responses to the corresponding queries
        """

        if setting == 'zero_shot_cot':
            return self._response_str2str_zero_shot_cot(query_list)
        else:
            return self._response_str2str(query_list)

    @torch.inference_mode(mode=True)
    def _response_str2str(self, query_list: List[str]) -> List[str]:
        assert isinstance(query_list, list) and isinstance(query_list[0], str)

        query_list = [self.system_prompt.format(question=query) for query in query_list]
        input_ids = self.tokenizer(query_list, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        n_seq = input_ids.shape[-1]

        output_ids = self.accelerator.unwrap_model(self.model).generate(input_ids=input_ids, 
                                         generation_config=self.gconfig
                                         )[..., n_seq:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return responses

    @torch.inference_mode(mode=True)
    def _response_str2str_zero_shot_cot(self, query_list: List[str]) -> List[str]:
        """
        Sample one response for each query, compatible with zero-shot, few-shot, and zero-shot cot setting
        Args:
            query_list (`List[str]`)
                List of queries
        Return:
            responses (`List[str]`)
                List of responses to the corresponding queries
        """
        assert isinstance(query_list, list) and isinstance(query_list[0], str)

        query_list = [self.system_prompt.format(question=query) + "Let's think step by step." for query in query_list]
        input_ids = self.tokenizer(query_list, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)
        n_seq = input_ids.shape[-1]

        output_ids = self.accelerator.unwrap_model(self.model).generate(input_ids=input_ids, 
                                         generation_config=self.gconfig
                                         )[..., n_seq:]
        responses = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return responses
    


class GPTAgent:
    def __init__(
        self, 
        model_id: str,
        model_name: str,
        model: PreTrainedModel, 
        system_prompt: str,
        gconfig: GenerationConfig,
        accelerator: Accelerator,
    ):
        self.model_id = model_id
        self.model_name = model_name
        self.model = model
        self.system_prompt = system_prompt
        self.gconfig = gconfig
        self.accelerator = accelerator

        self.stage = 3
        self.tokenizer = None
        self.collate_fn = None

    @classmethod
    def from_model_id(
        cls,
        model_id: str,
        generation_type: str = 'greedy',
        accelerator: Accelerator = None,
    ):
        model = model_id
        system_prompt = "{question}"
        gconfig = {
            "temperature": 0 if 'greedy' in generation_type else None
        }

        return cls(
            model_id=model_id,
            model_name=model_id,
            model=model,
            system_prompt=system_prompt,
            gconfig=gconfig,
            accelerator=accelerator,
        )

    def generate_from_dataset(
        self,
        query_list: List[str],
        batch_size: int,
        description: str = None,
        device_placement: bool = True,
        **kwargs,
    ) -> List[str]:
        output_dir = f'benchmark_eval/results/temp/{description}'
        os.makedirs(output_dir, exist_ok=True)
        with multiprocessing.Pool(processes=256) as pool:
            responses = pool.map(qgen_sample,
                            [(sample_idx, sample, os.path.join(output_dir, f'{sample_idx}.json'))
                                for sample_idx, sample in enumerate(query_list)])

        return responses
    

def qgen_sample(args):
    from gpt import GPT
    sample_idx, sample, output_path = args
    if os.path.exists(output_path):
        answer = json.load(open(output_path))
        if answer != '':
            return answer

    success = True
    MAX_RETRY=100
    cnt = 0
    while True:
        try:
            if cnt > MAX_RETRY:
                success = False
                break
            # Generate Q
            model = GPT(user_name='Arabic-eval', model_name="gpt-3.5-turbo", new_version='0.1.0')
            flag, answer = model.call(sample, args={'temperature': 0})
            assert flag and answer[:5] != "Error"
            break
        except Exception as e:
            print(f"{sample_idx} {e} Retry...")
            cnt += 1

    if success:
        json.dump(answer, open(output_path, "wt", encoding="utf-8"), ensure_ascii=False)
        print(f"[{sample_idx}] succeeded.")
    else:
        print(f"[{sample_idx}] failed.")

    return answer


def encoding_acegpt(text: str, tokenizer):
    return tokenizer.encode(text, add_special_tokens=False)


