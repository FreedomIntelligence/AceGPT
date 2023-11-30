#!/bin/bash

batch_size=1
model_ids="llama-raw-pretrained-sencondround-ckpt0-430000"

n_shot=5
accelerate launch \
    eval.py \
    --model_id ${model_ids[i]} \
    --batch_size ${batch_size} \
    --benchmark_name MMLUArabic \
    --setting few_shot \
    --n_shot ${n_shot} \
    --generation_type MMLU