from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from utils.period_stop import StoppingCriteriaSub
class NormalDecoding:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_decode_defend(self, full_prompt, max_new_tokens, device):

        all_input_stuff = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = all_input_stuff.input_ids.to(device)
        attention_mask = all_input_stuff.attention_mask.to(device)
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens)

        output_sentence = self.tokenizer.batch_decode(output)[0]

        return output_sentence
    
    def generate_decode_defend_guanaco(self, full_prompt, max_new_tokens, device):

        all_input_stuff = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = all_input_stuff.input_ids.to(device)
        attention_mask = all_input_stuff.attention_mask.to(device)

        # 29889 is the token of " ."
        period = 29889
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stop=period)])
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)

        output_sentence = self.tokenizer.batch_decode(output)[0]

        return output_sentence

    def generate_decode_normal(self, full_prompt, device):

        all_input_stuff = self.tokenizer(full_prompt, return_tensors="pt")
        input_ids = all_input_stuff.input_ids.to(device)
        attention_mask = all_input_stuff.attention_mask.to(device)
        output = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=500)

        output_sentence = self.tokenizer.batch_decode(output)[0]

        return output_sentence