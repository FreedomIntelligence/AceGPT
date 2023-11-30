# Evaluation Pipeline


This project is mainly for benchmark evaluation (`benchmark_eval`) and model output generation (`generation`, which is prepared for automatic evaluation). By default, you should use this project in SRIBD machines. Otherwises, you should config the model parameter path by yourself in `config.yaml`

For automatic evaluation, please refer to [GPTReview](https://github.com/FreedomIntelligence/GPTReview)


## Supporting Models

Models are indexed by their ids.

<table>
    <tr>
      <th align="center">Model Series</th>
      <th align="center">Model Id</th>
    </tr >
    <tr>
        <td>bloomz</td> 
        <td>bloomz-7b1-mt</td> 
    </tr>
    <tr>
        <td>phoenix</td> 
        <td>phoenix-inst-chat-7b</td> 
    </tr>
    <tr>
        <td>chimera</td> 
        <td>chimera-inst-chat-7b</td> 
    </tr>
    <tr>
        <td>huatuo</td> 
        <td>new_huatuo_230K</td> 
    </tr>
    <tr>
        <td rowspan="2">chatglm</td> 
        <td>chatglm-6b</td> 
    </tr>
    <tr>
        <td>doctor-glm</td> 
    </tr>
    <tr>
        <td>llama</td> 
        <td>llama-7b-hf</td> 
    </tr>
    <tr>
        <td rowspan="2">vicuna</td> 
        <td>vicuna-7b-v1.1</td> 
    </tr>
    <tr>
        <td>vicuna-13b-v1.1</td> 
    </tr>
    <tr>
        <td rowspan="2">llama-lora</td> 
        <td>huatuo-llama-med-chinese</td> 
    </tr>
    <tr>
        <td>guanaco-7b-leh-v2</td> 
    </tr>
</table>


## Usage
For usage, you should first config your `PROJECT_DIR` in `utils/config_utils.py`
```python
PROJECT_DIR='/home/xxx/LLM-eval-pipeline'
```


## Benchmark Evaluation
See `README.md` in the corresponding folder

```bash
cd benchmark_eval
```


## Model Output Generation
See `README.md` in the corresponding folder

```bash
cd generation
```


## Add new models
Write in two files to add new model: `config.yaml`, `utils/config_utils.py`
1. `config.yaml`: specify model parameter directory, prompt (the placeholder must be `{question}`), precision, and so on. For example,
    ```yaml
    phoenix:
        phoenix-inst-chat-7b:
            config_dir: /mntcephfs/data/med/zhanghongbo/general_pretrain/phoenix-inst-chat-7b
            chat: true
            prompt: "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.\n\nHuman: <s>{question}</s>Assistant: <s>"
            precision: 'fp16'
    ```

2. `utils/config_utils.py`: specify how to load model and tokenizer. Models and tokenizers are loaded via functions `load_model` and `load_tokenizer` respectively by default. You can also specify specific loading ways by defining your functions and config them into `get_special_tokenizer_loading()` and `get_special_model_loading()`. For example,
    ```python
    def get_special_tokenizer_loading():
        return {
            'vicuna': load_llama_tokenizer,
            'llama-lora': load_llama_tokenizer,
        }

    def get_special_model_loading():
        return {
            'chatglm': load_ChatGLM,
            'doctor-glm': load_DoctorGLM,
            'llama-lora': load_llama_lora,
        }
    ```

## Add new decoding hyperparameter configs
You can config decoding hyperparameter under `gconfig/` and save it as a json. For example,
```json
{
  "do_sample": true,
  "max_length": 2048,
  "min_length": 20,
  "min_new_tokens": 0,
  "temperature": 0.7,
  "top_k": 50,
  "top_p": 1.0,
  "repetition_penalty": 1.0,
  "transformers_version": "4.29.0.dev0"
}
```



