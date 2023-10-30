"""
Generates a stopping criteria that stops generation when one of the stop tokens is generated.
"""
import torch
from transformers import StoppingCriteria

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    """
    Generates a stopping criteria that stops generation when one of the stop tokens is generated.
    """
    def __init__(self,tokenizer,stop_list,device):
        self.stop_token_ids = self.stop_token_id_generator(tokenizer,stop_list, device)

    def stop_token_id_generator(self,tokenizer,stop_list, device):
        """
        generate tensor of stop token ids
        """
        stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
        stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
        return stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False
