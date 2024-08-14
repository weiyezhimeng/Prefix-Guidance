from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
import torch
class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stop):
        super().__init__()
        self.stop = stop

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        if self.stop == input_ids[0][-1]:
            return True
        return False
# Llama-2, .id is 29889
# period = 29889

