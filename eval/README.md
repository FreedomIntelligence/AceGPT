# Evaluation Pipeline
This project focuses on the AceGPT benchmark evaluation (`benchmark_eval`).

## Usage
To use this pipeline, follow the steps below:

### Add Model Path
1. Update the model path in `mconfig/config.yaml`.
2. In `config.yaml`, specify the model parameter directory, prompt (ensure the placeholder is `{question}`), stage, and precision. Here is an example:

   ```yaml
   acegpt:
     AceGPT-7B-base:
       config_dir: <Your model path>  # Change this to your model's path
       prompt: "{question}"
       stage: 1
       precision: 'fp16'

### Run Evaluation Script
To evaluate, for example, the EXAMS dataset, run the following script:

```sh script/EXAMS_few_shot_7b.sh```

the MMLUArabic dataset, run the following script:

```sh script/MMLUArabic_few_shot_7b.sh```

