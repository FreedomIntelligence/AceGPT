# AceGPT:  Aligning Large Language Models with Local (Arabic) Values

# ✨ Latest News
* update AceGPT models on hugging face.(2023.12.02)
# ⚡ Introduction

Welcome to the repository of AceGPT.

AceGPT achieved top performance among open-source Arabic language models in benchmark tests such as `Arabic Vicuna-80`, `Arabic AlpacaEval`, `Arabic MMLU`, `EXAMs` and our newly proposed benchmark `Arabic Cultural&Value Alignment`.

Here is a list of what has been released:
* The datasets we used for benchmark testing which were processed by ourselves, including`Arabic Vicuna-80`, `Arabic AlpacaEval`, `Arabic MMLU`, `EXAMs ` and `Arabic Cultural&Value Alignment`.
* The code for training and inferencing.
* The models we have trained, include AceGPT-7B, AceGPT-13B, AceGPT-7B-chat and AceGPT-13B-chat.

# 💭 Overview
In this paper, we present AceGPT, an open-source Large Language Model (LLM) tailored for the Arabic language. AceGPT not only addresses the unique syntactic intricacies of Arabic but also ensures cultural sensitivity and alignment with local values. Our methodology encompasses incremental pre-training on Arabic texts, supervised fine-tuning (SFT) using genuine Arabic instructions paired with native GPT-4 responses, and a novel reinforcement learning approach termed Reinforcement Learning with AI Feedback (RLAIF). This last method incorporates a reward model sensitive to local culture and values. Ultimately, our aim is to deliver an Arabic LLM that is both culturally aware and value-aligned, adeptly serving the diverse linguistic and practical needs of the Arabic-speaking community.

# 📚Data
## Benchmark Datsets
* We released benchmark datasets in [eval](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval) .  

  
* About ALUE, you can check [ALUE](https://www.alue.org/tasks) to get questions. 

 
* We have also released our results on benchmark datasets, you can check  [eval_results](https://github.com/FreedomIntelligence/AceGPT/tree/main/eval_results) if needed.  

# 👨‍⚕️ Model

## Model Access
| Model                | Backbone      | Link                                                                          |
|----------------------|---------------|-------------------------------------------------------------------------------|
| AceGPT-7B | LlaMA2 | [Model_Weigths](https://huggingface.co/FreedomIntelligence/AceGPT-7B) |
| AceGPT-13B     | LlaMA2  | [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-13B)      |
| AceGPT-7B-chat | LlaMA2  | [Model_Weigths](https://huggingface.co/FreedomIntelligence/AceGPT-7B-chat) |
| AceGPT-13B-chat     | LlaMA2  | [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-13B-chat)      |
|AceGPT-7B-chat-int4|LlaMA2| [Model Weights](https://huggingface.co/FreedomIntelligence/AceGPT-7b-chat-GPTQ) |



## 🧐Deploy

Firstly, you should install all required packages
```bash
pip install -r requirements.txt
```

To use the int4-quanted model, please refer to the [Ace-GPT-7B-Chat-int4](https://huggingface.co/FreedomIntelligence/AceGPT-7b-chat-GPTQ). But the performance of the int4 version has experienced some degradation, for a better user experience, please use the fp16 version.

For all base and chat models, you can proceed by running the following command:
```bash
python -m web_demo.py --model-name $model_dir
```
# 🤖 Limitations
Our model is primarily designed and trained to function as an AI assistant tailored for Arabic speakers. This specific design focus means that while it is optimized for generating responses to queries in Arabic, it may not produce satisfactory results for queries in other languages. Furthermore, while we have made significant advancements in the model's capabilities, it is essential to recognize its potential pitfalls. These include possible misuse, such as mishandling sensitive information, producing harmful content, perpetuating misinformation, or failing safety checks. We have not conducted an exhaustive safety check on the model, so users should exercise caution. We cannot overemphasize the need for responsible and judicious use of our model. Moreover, our evaluations predominantly relied on open-source data and the data we crafted. To achieve a more robust and comprehensive assessment, and to bolster the credibility of our findings, constructing an expansive evaluation set is imperative.

# 😀 Acknowledgement

We are aware that our works are inspired by the following works, including but not limited to

- Bloom: https://huggingface.co/bigscience/bloom
- Self-instruct: https://github.com/yizhongw/self-instruct
- LLMZoo: https://github.com/FreedomIntelligence/LLMZoo
- LlaMA：https://github.com/facebookresearch/llama
  
Without these, nothing could happen in this repository.


# Citation
```
@misc{huang2023acegpt,
      title={AceGPT, Localizing Large Language Models in Arabic}, 
      author={Huang Huang and Fei Yu and Jianqing Zhu and Xuening Sun and Hao Cheng and Dingjie Song and Zhihong Chen and Abdulmohsen Alharthi and Bang An and Ziche Liu and Zhiyi Zhang and Junying Chen and Jianquan Li and Benyou Wang and Lian Zhang and Ruoyu Sun and Xiang Wan and Haizhou Li and Jinchao Xu},
      year={2023},
      eprint={2309.12053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
We are from the School of Data Science, the Chinese University of Hong Kong, Shenzhen (CUHKSZ), the Shenzhen Research Institute of Big Data (SRIBD), and the King Abdullah University of Science and Technology (KAUST).



<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/AceGPT&type=Date&theme=dark" />
  <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=FreedomIntelligence/AceGPT&type=Date" />
  <img alt="Star History Chart" src="" />
</picture>

