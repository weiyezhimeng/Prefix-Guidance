def bert_judge_malicous(text, device, tokenizer_bert, model_bert):
    text_tokens = tokenizer_bert(text, return_tensors="pt")
    input_ids = text_tokens['input_ids'].to(device)
    attention_mask = text_tokens['attention_mask'].to(device)
    logits_user = model_bert(input_ids, attention_mask)[0]
    if logits_user[0] > logits_user[1]:
        return True
    else:
        return False