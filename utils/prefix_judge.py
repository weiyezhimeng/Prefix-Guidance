from utils.bert_judge import bert_judge_malicous
import time
def prefix_judge(prompt, prefix, decoder, max_new_tokens, device, tokenizer_bert, model_bert):
    # " I'm sorry". Space before " I'm sorry" cannot be ignored.
    full_prompt = prompt + prefix
    
    start_time = time.time()#============================

    response = decoder.generate_decode_defend(full_prompt, max_new_tokens, device)

    len_prompt = len(prompt)
    len_prefix = len(prefix)
    len_sapce = len(" ")
    len_start_token_space = len("<s>") + len(" ")

    assistant_response = response[len_start_token_space + len_prompt + len_prefix + len_sapce:]
    # print(assistant_response, flush=True)
    malicious_or_not = bert_judge_malicous(assistant_response, device, tokenizer_bert, model_bert)

    end_time = time.time()#================================
    time_to_generate_and_judge = end_time - start_time

    return malicious_or_not, time_to_generate_and_judge, response[len_start_token_space:]