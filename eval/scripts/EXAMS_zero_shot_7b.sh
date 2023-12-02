#!/bin/bash

#export CUDA_VISIBLE_DEVICES=0 
model_ids="AceGPT-7B-chat"
batch_size=1

accelerate launch \
        eval.py \
        --model_id ${model_ids} \
        --batch_size ${batch_size} \
        --benchmark_name EXAMS_Arabic \
        --setting zero_shot \
        --generation_type EXAMS