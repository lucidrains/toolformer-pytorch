from functools import partial
from collections import namedtuple

import torch
from torch import nn, einsum

from einops import rearrange, reduce
from toolformer_pytorch.palm import PaLM

from beartype import beartype
from beartype.typing import Callable

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def all_contains_id(t: torch.Tensor, token_id: int):
    mask = t == token_id
    return mask.any(dim = -1).all()

# the main contribution of the paper is simply the filtering equations presented in section 2

def default_weight_fn(t):
    # following the formula in section 4.1 - however, not sure what w_s is in the denominator
    # if t stands for each timestep, this would also mean within 5 tokens it would diminish to 0?
    return (1. - t * 0.2).clamp(min = 0.)

def get_pred_prob(token_ids, logits):
    logits = logits[:, :-1]             # logits of each token...    (omit last logit)
    token_ids = token_ids[:, 1:]        # predicts the next token id (omit first token id)

    token_ids = rearrange(token_ids, 'b n -> b n 1')
    probs = logits.softmax(dim = -1)
    correct_token_id_pred_prob = probs.gather(-1, token_ids)
    return rearrange(correct_token_id_pred_prob, 'b n 1 -> b n')

def get_arange_start_at_token_id(token_ids, token_id, pad_id = -1):
    arange = (token_ids == token_id).cumsum(dim = -1)
    before_token_mask = arange == 0
    arange -= 1
    return arange    

FilteredResults = namedtuple('FilteredResults', [
    'selected_indices',
    'selected_mask',
    'filtered_tokens',
    'filtered_tokens_without_api_response',
    'filtered_tokens_with_api_response'
])

@beartype
def filter_tokens_with_api_response(
    model: nn.Module,                              # the language model should accept the token ids below and return the logits in shape (batch, seq, num tokens)
    *,
    tokens: torch.Tensor,                          # token ids (batch, seq) of the original passage, without api calls
    tokens_without_api_response: torch.Tensor,     # token ids (batch, seq) of the passage, but with the api call (but without a response filled in) - <api>tool1(x, y)</api>
    tokens_with_api_response: torch.Tensor,        # token ids (batch, seq) of the passage with api call and the response - <api>tool1(x, y) → {response}</api>
    api_start_token_id: int,                       # token id of the <api> tag
    api_end_token_id: int,                         # token id of the </api> tag
    filter_threshold: float = 1.,                  # the threshold at which to accept the sampled api call (tokens_with_api_response) for fine-tuning
    weighting_fn: Callable = default_weight_fn     # weighting function
) -> FilteredResults:

    # validations

    assert all([*map(lambda t: t.dtype == torch.long, (tokens, tokens_with_api_response, tokens_without_api_response))])

    assert all_contains_id(tokens_without_api_response, api_start_token_id)
    assert all_contains_id(tokens_without_api_response, api_end_token_id)

    assert all_contains_id(tokens_with_api_response, api_start_token_id)
    assert all_contains_id(tokens_with_api_response, api_end_token_id)

    # get all the logits

    with torch.no_grad():
        model.eval()
        logits, logits_without_api_response, logits_with_api_response = map(model, (tokens, tokens_with_api_response, tokens_without_api_response))

    # derive all predicted prob of the actual next token id in sequence

    probs                       = get_pred_prob(tokens, logits)
    probs_without_api_response  = get_pred_prob(tokens_without_api_response, logits_without_api_response)
    probs_with_api_response     = get_pred_prob(tokens_with_api_response, logits_with_api_response)

    # derive the weighting

    t_without_api_response = get_arange_start_at_token_id(tokens_without_api_response, api_end_token_id)
    t_with_api_response = get_arange_start_at_token_id(tokens_with_api_response, api_end_token_id)

    t_without_api_response = t_without_api_response[:, :-1]
    t_with_api_response = t_with_api_response[:, :-1]

    weight_without_api_response = weighting_fn(t_without_api_response)
    weight_with_api_response = weighting_fn(t_with_api_response)

    weight_without_api_response = weight_without_api_response.masked_fill(t_without_api_response == -1, 0.)
    weight_with_api_response = weight_without_api_response.masked_fill(t_with_api_response == -1, 0.)

    # deriving the weighting for the original passage is more tricky
    # would need to start counting up from <api> start token location

    t = get_arange_start_at_token_id(tokens_without_api_response, api_start_token_id)
    t = t[:, 1:]

    weight = weighting_fn(t) # shift to the left by one since <api> does not exist in the original sequence
    weight = weight.masked_fill(t == -1, 0.)

    # get the loss L for all three types of sequences

    loss = (-weight * log(probs)).sum(dim = -1)
    loss_without_api_response = (-weight_without_api_response * log(probs_without_api_response)).sum(dim = -1)
    loss_with_api_response = (-weight_with_api_response * log(probs_with_api_response)).sum(dim = -1)

    # calculate the main formula in the paper

    # loss+ = loss with api response
    # loss- = min(loss without api response, loss without api at all)

    loss_plus = loss_with_api_response
    loss_minus = torch.minimum(loss_without_api_response, loss)

    selected_mask = (loss_minus - loss_plus) >= filter_threshold

    # now we can select and return the entries that survived the filtering stage
    # also returning the selected indices of the batch being processed
    # for finetuning the model into toolformer

    batch = tokens.shape[0]
    indices = torch.arange(batch, device = tokens.device)

    selected_indices = indices[selected_mask]

    ret = FilteredResults(
        selected_indices,
        selected_mask,
        tokens[selected_mask],
        tokens_without_api_response[selected_mask],
        tokens_with_api_response[selected_mask]
    )

    return ret

# classes

@beartype
class Toolformer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        tool: Callable,
        teach_tool_prompt: str
    ):
        super().__init__()
        self.model = model
        self.tool = tool
        self.teach_tool_prompt = teach_tool_prompt

    def forward(self):
        raise NotImplementedError
