# AceGPT:  Aligning Large Language Models with Local (Arabic) Values

# ✨ Latest News


# ⚡ Introduction

Welcome to the repository of AceGPT.

AceGPT achieved top performance among open-source Arabic language models in benchmark tests such as `Arabic Vicuna-80`, `Arabic AlpacaEval`, `Arabic MMLU`, `EXAMs` and our newly proposed benchmark `Arabic Cultural&Value Alignment`.

Here is a list of what has been released:
* The datasets we used for benchmark testing which was processed by ourselves, including`Arabic Vicuna-80`, `Arabic AlpacaEval`, `Arabic MMLU`, `EXAMs ` and `Arabic Cultural&Value Alignment`.
* The code for training and inferencing.
* The models we have trained, including AceGPT-7B, AceGPT-13B, AceGPT-7B-chat, AceGPT-13B-chat.

# 💭 Overview
In this paper, we present AceGPT, an open-source Large Language Model (LLM) tailored for the Arabic language. AceGPT not only addresses the unique syntactic intricacies of Arabic but also ensures cultural sensitivity and alignment with local values. Our methodology encompasses incremental pre-training on Arabic texts, supervised fine-tuning (SFT) using genuine Arabic instructions paired with native GPT-4 responses, and a novel reinforcement learning approach termed Reinforcement Learning with AI Feedback (RLAIF). This last method incorporates a reward model sensitive to local culture and values. Ultimately, our aim is to deliver an Arabic LLM that is both culturally-aware and value-aligned, adeptly serving the diverse linguistic and practical needs of the Arabic-speaking community.

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
| AceGPT-7B-chat | LlaMA2  | [Model_Weigths](https://huggingface.co/FreedomIntelligence/AceGPT-7B-chat) |
| AceGPT-13B-chat     | LlaMA2  | [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-13B-chat)      |




# 😀 Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- Bloom: https://huggingface.co/bigscience/bloom
- Self-instruct: https://github.com/yizhongw/self-instruct
- LLMZoo: https://github.com/FreedomIntelligence/LLMZoo
- LlaMA：https://github.com/facebookresearch/llama
  
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
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/AceGPT&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/AceGPT&type=Date" />
  <img alt="Star History Chart" src="" />
</picture>

