from transformers import AutoTokenizer
from transformers import pipeline
import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
import re
import argparse 
import gradio as gr
from threading import Thread

def load_model(model_name, device, num_gpus, precise='fp16'):
    if device == "cuda":
        kwargs = {"torch_dtype": torch.float32}
        if precise == 'int8':
            kwargs['load_in_8bit']=True
        if num_gpus == "auto":
            kwargs["device_map"] = "auto"
        else:
            num_gpus = int(num_gpus)
            if num_gpus != 1:
                kwargs.update({
                    "device_map": "auto",
                    "max_memory": {i: "13GiB" for i in range(num_gpus)},
                })
    elif device == "cpu":
        kwargs = {}
    else:
        raise ValueError(f"Invalid device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="right", use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True, **kwargs)
    if precise=='fp16':
        model = model.half()
    
    if device == "cuda" and num_gpus == 1:
        model.cuda()

    return model, tokenizer
    
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False
    
    
def main(args):
    model, tokenizer = load_model(args.model_name, args.device, args.num_gpus, args.precise)
    model = model.eval()
    
    prompt_dict = {
        'AceGPT': """[INST] <<SYS>>\nأنت مساعد مفيد ومحترم وصادق. أجب دائما بأكبر قدر ممكن من المساعدة بينما تكون آمنا.  يجب ألا تتضمن إجاباتك أي محتوى ضار أو غير أخلاقي أو عنصري أو جنسي أو سام أو خطير أو غير قانوني. يرجى التأكد من أن ردودك غير متحيزة اجتماعيا وإيجابية بطبيعتها.\n\nإذا كان السؤال لا معنى له أو لم يكن متماسكا من الناحية الواقعية، اشرح السبب بدلا من الإجابة على شيء غير صحيح. إذا كنت لا تعرف إجابة سؤال ما، فيرجى عدم مشاركة معلومات خاطئة.\n<</SYS>>\n\n""",
    }
    
    
    # all role
    role_dict = {
        'AceGPT':['[INST]','[/INST]'],
    }
    
    # all start and end token
    se_tok_dict = {
        'AceGPT':["","</s>"],
    }
    
    
    
    def format_message(query, history, max_src_len):
        if not history:
            return f"""{prompt_dict["AceGPT"]}{query} {role_dict["AceGPT"][1]}"""
        else:
            prompt = prompt_dict["AceGPT"]
            filter_historys = []
            memory_size = len(prompt) + len(query)
            for rev_idx in range(len(history) - 1, -1, -1):
                this_turn_len = len(tokenizer.encode(history[rev_idx][0]) + tokenizer.encode(history[rev_idx][1]))
                if memory_size + this_turn_len > max_src_len:
                    break
                filter_historys.append(history[rev_idx])
                memory_size += this_turn_len
            filter_historys.reverse()
            for i, (old_query, response) in enumerate(filter_historys):
                prompt += f'{old_query} {role_dict["AceGPT"][1]}{response}{se_tok_dict["AceGPT"][1]}{role_dict["AceGPT"][0]} '
            prompt += f'{query} {role_dict["AceGPT"][1]}'
            return prompt
        

    def get_llama_response(message: str, history: list) -> str:
        """
        Generates a conversational response from the Llama model.
    
        Parameters:
            message (str): User's input message.
            history (list): Past conversation history.
    
        Returns:
            str: Generated response from the Llama model.
        """

        temperature=0.5
        max_new_tokens = 768
        content_len = 2048
        stop = StopOnTokens()
        max_src_len = content_len-max_new_tokens-8
        prompt = format_message(message, history, max_src_len)

        model_inputs  = tokenizer(prompt, return_tensors="pt").to("cuda")
        streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.95,
            top_k=1000,
            temperature=temperature,
            num_beams=1,
            stopping_criteria=StoppingCriteriaList([stop])
            )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()

        partial_message  = ''
        for new_token in streamer:
            if new_token != '</s>':
                partial_message += new_token
                yield partial_message

    
    gr.ChatInterface(get_llama_response, chatbot=gr.Chatbot(rtl=True)).queue().launch(share=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="FreedomIntelligence/AceGPT-7B-chat")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--precise", type=str, choices=["fp16", "int8"], default="fp16")

    args = parser.parse_args()
    main(args)
