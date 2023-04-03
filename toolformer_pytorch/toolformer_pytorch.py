from functools import partial
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum

from einops import rearrange, reduce
from toolformer_pytorch.palm import PaLM

from beartype import beartype
from beartype.typing import Callable, Optional

from tqdm import tqdm

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# tensor helpers

def log(t, eps = 1e-20):
    return t.clamp(min = eps).log()

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, eps = 1e-10):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / max(temperature, eps)) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.9):
    k = math.ceil((1 - thres) * logits.shape[-1])
    val, indices = torch.topk(logits, k)
    probs = torch.full_like(logits, -torch.finfo(logits.dtype).max)
    probs.scatter_(1, indices, val)
    return probs

def all_contains_id(t: torch.Tensor, token_id: int):
    mask = t == token_id
    return mask.any(dim = -1).all()

# sampling api related functions
# they do greedy sampling, but encourage sampling api calls by auto-selecting <api> when that token is in the top k = 10

@torch.no_grad()
def sample(
    model: nn.Module,
    *,
    seq_len,
    api_start_token_id,
    select_api_start_id_top_k = 10,
    prime: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    batch_size = 1,
    eos_token_id = None,
    sos_token_id = 1,
    temperature = 0.,
    pad_id = 0
):
    device = next(model.parameters()).device
    max_seq_len = seq_len + 1

    # prime

    if exists(prime):
        batch_size, prime_length = prime.shape
    else:
        prime_length = 1
        prime = torch.full((batch_size, 1), sos_token_id, device = device, dtype = torch.long)

    prime = prime.to(device)

    # sampling positions - different sequences have different cursors

    positions = default(positions, torch.zeros((batch_size,), device = device, dtype = torch.long))
    assert (positions <= prime_length).all() and (positions < max_seq_len).all(), 'all positions must be less then initial prime length as well as the total sequence length + 1 (plus one for noop if one sequence finished sampling before the other)'

    # eval model

    model.eval()

    # lengthen the prime to the entire sequence length

    remain_iterations = seq_len - prime_length
    output = F.pad(prime, (max_seq_len - prime_length, 0), value = 0.)

    batch_indices = torch.arange(batch_size, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    position_indices = rearrange(positions, 'b -> b 1')

    # determine the <api> token mask, for making sure api is called only once, masking out logit to prevent it from being selected for those rows which already contains an <api> token

    api_token_mask = None # lazily created, since do not know logit dimensions

    def create_api_token_mask(num_tokens, api_start_token_id):
        mask = torch.zeros((1, 1, num_tokens), dtype = torch.bool)
        assert api_start_token_id < num_tokens
        mask[..., api_start_token_id] = True
        return mask

    # start iterating

    for iteration in tqdm(range(remain_iterations)):
        logits = model(output)
        last_logits = logits[batch_indices, position_indices]

        if not exists(api_token_mask):
            num_tokens = last_logits.shape[-1]
            api_token_mask = create_api_token_mask(num_tokens, api_start_token_id)
            api_token_mask = api_token_mask.to(device)

        api_called = (output == api_start_token_id).any(dim = -1)

        # this will ensure that each batch token sequence will have at most one <api> token

        logit_mask = api_token_mask & rearrange(api_called, 'b -> b 1 1')
        last_logits = last_logits.masked_fill(logit_mask, -torch.finfo(last_logits.dtype).max)

        # greedy sample (but could be made non-greedy)

        sampled = gumbel_sample(last_logits, temperature = temperature)

        # for those sequences without an api call, if the api_start_token_id is within top k (set to 10 in paper) of logits, just auto-select

        # seems to be an important hack in the paper
        # it seems like this paper will take a lot more follow up research to be viable

        top_token_ids = last_logits.topk(select_api_start_id_top_k, dim = -1).indices
        has_api_token_in_topk = (top_token_ids == api_start_token_id).any(dim = -1)
        should_auto_select_api_token = has_api_token_in_topk & ~rearrange(api_called, 'b -> b 1')

        sampled = sampled.masked_fill(should_auto_select_api_token, api_start_token_id)

        # set the sampled tokens at the right curosr positions

        output[batch_indices, position_indices] = sampled

        # increment positions

        position_indices += 1
        position_indices.clamp_(max = seq_len) # noop if one sequence is further along and near the end

        # if using <eos> tokens, look for all sequences having it and terminate, also anything after <eos> will be padded

        if exists(eos_token_id):
            eos_mask = (output == eos_token_id)
            all_rows_have_eos = eos_mask.any(dim = -1).all()

            if all_rows_have_eos:
                keep_mask = eos_mask.cumsum(dim = -1) == 0
                keep_mask = F.pad(keep_mask, (1, 0), value = True)
                output = output.masked_fill(~keep_mask, pad_id)
                break

    # remove the last token in output (use as noop placeholder)

    output = output[:, :-1]

    return output

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

def get_arange_start_at_token_id(
    token_ids: torch.Tensor,
    token_id: int,
    pad_id = -1
):
    is_token_id_mask = token_ids == token_id
    arange = (is_token_id_mask.cumsum(dim = -1) > 0).cumsum(dim = -1)
    before_token_mask = arange == 0
    arange = arange - 1
    arange = arange.masked_fill(before_token_mask, pad_id)
    return arange

def weight_and_mask(
    token_ids: torch.Tensor,
    token_id: int,
    pad_id = -1,
    weighting_fn: Callable = default_weight_fn
):
    t = get_arange_start_at_token_id(token_ids, token_id, pad_id)
    weights = weighting_fn(t)
    return weights.masked_fill(weights == pad_id, 0.)

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

    weight_and_mask_fn = partial(weight_and_mask, weighting_fn = weighting_fn)

    # derive the weighting

    weight_without_api_response = weight_and_mask_fn(tokens_without_api_response[:, :-1], api_end_token_id)
    weight_with_api_response = weight_and_mask_fn(tokens_with_api_response[:, :-1], api_end_token_id)

    # deriving the weighting for the original passage is more tricky
    # would need to start counting up from <api> start token location

    weight = weight_and_mask_fn(tokens_without_api_response[:, 1:], api_start_token_id) # shift to the left by one since <api> does not exist in the original sequence

    # get the loss L for all three types of sequences

    def loss_fn(weight, probs):
        return (weight * -log(probs)).sum(dim = -1)

    loss = loss_fn(weight, probs)
    loss_without_api_response = loss_fn(weight_without_api_response, probs_without_api_response)
    loss_with_api_response = loss_fn(weight_with_api_response, probs_with_api_response)

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
