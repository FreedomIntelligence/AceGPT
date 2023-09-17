# AceGPT:  Aligning Large Language Models with Local (Arabic) Values

# ✨ Latest News


# ⚡ Introduction

Welcome to the repository of AceGPT.

AceGPT achieved top performance among open-source Arabic language models in benchmark tests such as `Arabic Vicuna-80`, `Arabic MMLU`, `EXAMs`, `AlpacaEval` and our newly proposed benchmark `Arabic cultural&value alignment`.

Here is a list of what has been released:
* The datasets we used for benchmark testing which was processed by ourselves, including`Arabic Vicuna-80`, `Arabic MMLU`, `EXAMs `, `AlpacaEval` and `Arabic cultural&valuealignment`.
* The code for training and inferencing.
* The models we have trained, including AceGPT-7B, AceGPT-13B, AceGPT-chat-7B, AceGPT-chat-13B.

# 💭 Overview
We introduced AceGPT, an open-source LLM, to address the unique syntactic and cultural characteristics of the Arabic language, including cultural sensitivity and alignment with local values. We proposed a comprehensive solution to address the aforementioned challenges, which includes incremental pre-training using Arabic text, supervised fine-tuning (SFT) with actual Arabic instructions and native GPT-4 responses, and reinforcement learning with human feedback (RLHF) that takes into account local culture and values through a reward model.Our objective is to train a culturally-aware and value-aligned Arabic LLM that caters to the diverse language and application needs of the Arabic-speaking community.

#📚Data
## Benchmark Datsets
* We released benchmark datasets in [eval](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval) .  

  
* About ALUE, you can check [ALUE](https://www.alue.org/tasks) to get questions. 

 
* We have also released our results on benchmark datasets, you can check  [eval_results](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval_results) if needed.  

# 🚀 Training
```
python finetuning.py
```
# 🧐 Inferencing
```
python generate.py
```
# 👨‍⚕️ Model

## Model Access
| Model                | Backbone      | Link                                                                          |
|----------------------|---------------|-------------------------------------------------------------------------------|
| AceGPT-7B | LlaMA2 | [Model_Weigths](https://huggingface.co/FreedomIntelligence/AceGPT-7B) |
| AceGPT-13B     | LlaMA2  | [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-13B)      |
| AceGPT-chat-7B | LlaMA2  | [Model_Weigths](https://huggingface.co/FreedomIntelligence/AceGPT-chat-7B) |
| AceGPT-chat-13B     | LlaMA2  | [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-chat-13B)      |




# 😀 Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- Bloom: https://huggingface.co/bigscience/bloom
- Self-instruct: https://github.com/yizhongw/self-instruct
- LLMZoo: https://github.com/FreedomIntelligence/LLMZoo
  
Without these, nothing could happen in this repository.


# Citation
```
@misc{fan2023grammargpt,
      title={AceGPT: Aligning Large Language Models with Local (Arabic) Values}, 
      author={},
      year={2023},
      eprint={},
      archivePrefix={},
      primaryClass={}
}
```
We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ), and the Shenzhen Research Institute of Big Data (SRIBD).


<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/GrammarGPT&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/GrammarGPT&type=Date" />
  <img alt="Star History Chart" src="" />
</picture>
