# Prefix Guidance: A Steering Wheel for LLMs to Defend Against Jailbreak Attacks
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import RobertaTokenizer, RobertaModel
import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import json
from tqdm import tqdm
import random
from utils.string_utils import PromptManager, load_conversation_template
from datasets import load_dataset, concatenate_datasets
import argparse
from utils.decoding import NormalDecoding
from utils.bert_judge import bert_judge_malicous
from utils.prefix_judge import prefix_judge
import datetime

class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = model_bert
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[1]
        logits = self.classifier(last_hidden_state_cls)
        return logits

device = "cuda:0"
tokenizer_path_bert = "roberta-base"
tokenizer_bert = RobertaTokenizer.from_pretrained(tokenizer_path_bert, padding_side='right')
model_bert = torch.load("model_classify/classify_roberta.bin")

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vicuna")
    parser.add_argument("--path", type=str, default="vicuna-7b-v1.5")
    parser.add_argument("--attack_method", type=str, default="renellm")
    parser.add_argument("--prefix_llama", type=str, default=" I")
    parser.add_argument("--prefix_vicuna", type=str, default=" I'm sorry, but I cannot")
    parser.add_argument("--prefix_guanaco", type=str, default=" I'm sorry, but I cannot encourage anyone to")
    parser.add_argument("--max_new_tokens_for_judge", type=int, default=50)

    return parser.parse_args()

args = get_args()

device = "cuda:0"
tokenizer_path = model_path = args.path
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

result = []

if args.attack_method == "PAIR" or args.attack_method == "AutoDAN" or args.attack_method == "GCG" or args.attack_method == "DeepInception" or args.attack_method == "renellm":

    with open("dataset/malicious_prompt/" + args.attack_method + ".json", 'r') as file:
        data = json.load(file)

    for item in tqdm(data, desc='Processing records'):

        meta_result = {
        "time_to_generate_and_judge": 0,
        "malicious_or_not": False,
        "output": ""
        }

        if args.attack_method != "DeepInception" and args.attack_method != "renellm":
            if item["target-model"] != args.model:
                continue
        if args.attack_method == "PAIR" or args.attack_method == "DeepInception" or args.attack_method == "renellm":
            whitebox_attacker = False
        elif args.attack_method == "AutoDAN" or args.attack_method == "GCG":
            whitebox_attacker = True

        conversation = load_conversation_template(args.model)
        input_manager = PromptManager(tokenizer=tokenizer, conv_template=conversation, instruction=item["prompt"], whitebox_attacker=whitebox_attacker)
        prompt = input_manager.get_prompt()
        print(prompt, flush=True)
        print("===============================")
        
        len_prompt = len(prompt)
        len_sapce = len(" ")
        len_start_token_space = len("<s>") + len(" ")

        decoder = NormalDecoding(model, tokenizer)
        if args.model == "guanaco":
            meta_result["malicious_or_not"], meta_result["time_to_generate_and_judge"], response_temp = prefix_judge(prompt, args.prefix_guanaco, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)
        elif args.model == "llama2":
            meta_result["malicious_or_not"], meta_result["time_to_generate_and_judge"], response_temp = prefix_judge(prompt, args.prefix_llama, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)
        elif args.model == "vicuna":
            meta_result["malicious_or_not"], meta_result["time_to_generate_and_judge"], response_temp = prefix_judge(prompt, args.prefix_vicuna, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)

        if meta_result["malicious_or_not"]:
            meta_result["output"] = decoder.generate_decode_normal(response_temp, device)[len_start_token_space + len_prompt + len_sapce:]
        else:
            meta_result["output"] = decoder.generate_decode_normal(prompt, device)[len_start_token_space + len_prompt + len_sapce:]
        
        print(meta_result["output"], flush=True)
        print("========================================")
        result.append(meta_result)

elif args.attack_method == "Just-Eval":
    data = load_dataset('re-align/just-eval-instruct', split="test")
    for item in tqdm(data):
        output_formatted = {
                "id": item["id"],
                "instruction": item["instruction"],
                "source_id": item['source_id'],
                "dataset": item['dataset'],
                "output": "",
                "generator": args.model+f'_Just-Eval_PG',
                "time_cost": 0,
                "datasplit": "just_eval"
            }
        conversation = load_conversation_template(args.model)
        input_manager = PromptManager(tokenizer=tokenizer, conv_template=conversation, instruction=item["instruction"])
        prompt = input_manager.get_prompt()

        len_prompt = len(prompt)
        len_sapce = len(" ")
        len_start_token_space = len("<s>") + len(" ")
        
        malicious_or_not = False
        decoder = NormalDecoding(model, tokenizer)
        if args.model == "guanaco":
            malicious_or_not, output_formatted["time_cost"], response_temp = prefix_judge(prompt, args.prefix_guanaco, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)
        elif args.model == "llama2":
            malicious_or_not, output_formatted["time_cost"], response_temp = prefix_judge(prompt, args.prefix_llama, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)
        elif args.model == "vicuna":
            malicious_or_not, output_formatted["time_cost"], response_temp = prefix_judge(prompt, args.prefix_vicuna, decoder, args.max_new_tokens_for_judge, device, tokenizer_bert, model_bert)
        
        if malicious_or_not:
            output_formatted["output"] = decoder.generate_decode_normal(response_temp, device)[len_start_token_space + len_prompt + len_sapce:]
        else:
            output_formatted["output"] = decoder.generate_decode_normal(prompt, device)[len_start_token_space + len_prompt + len_sapce:]
        
        print(output_formatted, flush=True)
        result.append(output_formatted)

else:
    raise ValueError("attack method must be DeepInception, PAIR, AutoDAN or GCG")



# save result
current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y%m%d_%H%M%S")
if args.model == "guanaco":
    result_name = "{}_{}_{}_{}.json".format(args.model, args.attack_method, args.prefix_guanaco, time_str)
elif args.model == "vicuna":
    result_name = "{}_{}_{}_{}.json".format(args.model, args.attack_method, args.prefix_vicuna, time_str)
elif args.model == "llama2":
    result_name = "{}_{}_{}_{}.json".format(args.model, args.attack_method, args.prefix_llama, time_str)
json_result = json.dumps(result, indent=4)
with open("output/" + result_name, 'w') as json_file:
    json_file.write(json_result)