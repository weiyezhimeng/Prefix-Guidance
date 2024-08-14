import torch
from fastchat import model
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
import random
import pandas as pd

def process_text(text, model_main, model_path, tokenizer, device, prefix):
    # print(text)
    conversation = model.get_conversation_template(model_path)
    conversation.append_message(conversation.roles[0], text)
    conversation.append_message(conversation.roles[1], f"")
    full_prompt = conversation.get_prompt() + prefix
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device)
    output = model_main.generate(input_ids, max_new_tokens=50)

    len_sys_sorry = len("<s> " + full_prompt)
    len_sys = len_sys_sorry - len(prefix)
    output_sentence = tokenizer.batch_decode(output)[0][len_sys:]

    return output_sentence

def deny_try_csv(csv_file, csv_path, tokenizer_path, model_path):
    # loading the model
    device = "cuda:0"
    model_main = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # read the csv file
    df = pd.read_csv(csv_file)
    
    # result that need to be saved in the disk
    total_result = []

    # model response to text
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing rows'):
        response = process_text(row["text"], model_main, model_path, tokenizer, device)
        total_result.append(response)

    llm_responses_df = pd.DataFrame(total_result, columns=['response'])

    # save the file
    llm_responses_df.to_csv(csv_path, index=False)

def deny_try_json(json_file, json_path, tokenizer_path, model_path):
    device = "cuda:0"
    model_main = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    sample_size = 1550
    sample_data = random.sample(data, sample_size)

    total_result = []

    for i, item in tqdm(list(enumerate(sample_data))):
        if item["input"] == "":
            prompt_normal = item["instruction"]
        else:
            prompt_normal = item["instruction"] + "/n/n" + item["input"]

        response = process_text(prompt_normal, model_main, model_path, tokenizer, device)
        total_result.append(response)

    llm_responses_df = pd.DataFrame(total_result, columns=['response'])

    # save the file
    llm_responses_df.to_csv(json_path, index=False)