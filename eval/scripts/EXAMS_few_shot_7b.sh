#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0 
model_ids="AceGPT-7B-base"
batch_size=1
n_shot=5

accelerate launch \
        eval.py \
        --model_id ${model_ids} \
        --batch_size ${batch_size} \
        --benchmark_name EXAMS_Arabic \
        --setting few_shot \
        --n_shot ${n_shot} \
        --generation_type EXAMS
