import re
import numpy as np
from tqdm import tqdm
from transformers import GenerationConfig, AutoConfig, AutoModelForCausalLM, AutoTokenizer, AutoModel
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Tuple, Sequence, List, Union
from transformers import PreTrainedModel, PreTrainedTokenizerFast, PreTrainedTokenizer
import os
import torch
import json

config = OmegaConf.load(f'mconfigs/config.yaml')

def get_special_loading():  # lookup the specified method to load model and tokenizer. it can be indexed by either model_id (prior) or model_name
    return {
        'llama': load_llama,
        'acegpt': load_acegpt,
    }


def read_config_by_model_id(model_id: str) -> Tuple[str, DictConfig]:
    """Read the config in file by `model_id`, and Also Return the corresponding model_name"""
    if model_id is None:
        return
    
    if not isinstance(model_id, str):
        raise ValueError(f"model_id should be str, but it is {model_id}")

    for model_name in config.keys():
        if model_id in config[model_name]:
            return model_name, config[model_name][model_id]
    raise NotImplementedError(f"{model_id}, this model id isn't implemented in the `config.yaml`")

def load_stage(model_id: str) -> int:
    """Load and Return stage by `model_id`"""
    model_name, model_config = read_config_by_model_id(model_id)
    return model_config['stage']

def load_system_prompt(model_id: str) -> str:
    """Load and Return prompt by `model_id`"""
    model_name, model_config = read_config_by_model_id(model_id)
    return model_config['prompt']


def load_gconfig(generation_type: str='sample') -> GenerationConfig:
    """Load and Return GenerationConfig for the corresponding `model_id` and `generation_type`"""
    return GenerationConfig.from_pretrained("./gconfig", 
                                            config_file_name=f'{generation_type}.json')


def load_model_and_tokenizer(model_id: str) -> Tuple[PreTrainedModel, Union[PreTrainedTokenizerFast, PreTrainedTokenizer]]:
    """Load and Return model and tokenizer by `model_id`"""
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']
    
    precision = model_config['precision']
    assert precision in ('fp16', 'fp32'), 'Only supports fp16 and fp32 for now'

    special_loading = get_special_loading()
    loading_fn = special_loading.get(model_id, special_loading.get(model_name, None))
    if loading_fn:
        return loading_fn(model_id)

    if precision == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True,
                                                     trust_remote_code=True)
    elif precision == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(config_dir, low_cpu_mem_usage=True, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', 
                                              trust_remote_code=True)

    return model_name, model, tokenizer








def load_llama(model_id):
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']
    precision = model_config['precision']

    if precision == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    elif precision == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(config_dir, low_cpu_mem_usage=True, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', use_fast=False, trust_remote_code=True)
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
    })

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens(dict(pad_token="<unk>"))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    return model_name, model, tokenizer




def load_acegpt(model_id):
    model_name, model_config = read_config_by_model_id(model_id)
    config_dir = model_config['config_dir']
    precision = model_config['precision']

    if precision == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(config_dir, torch_dtype=torch.float16, low_cpu_mem_usage=True, trust_remote_code=True)
    elif precision == 'fp32':
        model = AutoModelForCausalLM.from_pretrained(config_dir, low_cpu_mem_usage=True, trust_remote_code=True)

    tokenizer = AutoTokenizer.from_pretrained(config_dir, padding_side='left', use_fast=False, trust_remote_code=True)
    if 'llama' in model_id.lower():
        tokenizer.add_special_tokens({
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
        })

        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(dict(pad_token="<unk>"))

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
    return model_name, model, tokenizer






