import re

from functools import partial, wraps
from collections import namedtuple

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from einops import rearrange, reduce

from toolformer_pytorch.palm import PaLM
from toolformer_pytorch.optimizer import get_optimizer
from toolformer_pytorch.prompts import DEFAULT_PROMPT_INPUT_TAG

from beartype import beartype
from beartype.typing import Callable, Optional, Union, List, Tuple

from tqdm import tqdm
from x_clip.tokenizer import tokenizer

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def identity(t):
    return t

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def try_except(fn, callback = identity):
    @wraps(fn)
    def inner(*args):
        try:
            return fn(*args)
        except Exception as e:
            return callback(e)
    return inner

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

def find_indices_of(t: torch.Tensor, token_id: int, occurrence = 1):
    assert occurrence > 0
    mask = (t == token_id)

    has_occurred = mask.cumsum(dim = -1)
    has_occurred = F.pad(has_occurred, (1, 0), value = 0.)

    return (has_occurred < occurrence).sum(dim = -1).long()

# invoking api call functions

def is_valid_string(s):
    return exists(re.fullmatch(r"'[^']*'|\"[^\"]*\"", s))

def is_valid_integer(s):
    return exists(re.fullmatch(r"[+-]?\d+", s))

def is_valid_float(s):
    return exists(re.fullmatch(r"[+-]?\d+(\.\d+)?", s))

def parse_param(s: str) -> Optional[Union[int, float, str]]:
    if is_valid_string(s):
        return str(s)
    elif is_valid_integer(s):
        return int(s)
    elif is_valid_float(s):
        return float(s)

    return None

@beartype
def replace_fn(
    registry: dict[str, Callable],
    matches,
    delimiter = '→'
):
    orig_text = matches.group(0)

    text_without_end_api_token = matches.group(1)
    end_api_token = matches.group(4)
    function_name = matches.group(2)

    # unable to find function in registry

    if function_name not in registry:
        return orig_text

    fn = registry[function_name]

    params = matches.group(3).split(',')
    params = list(map(lambda s: s.strip(), params))
    params = list(filter(len, params))
    params = list(map(parse_param, params))

    # if any of the parameters are not parseable, return

    if any([(not exists(p)) for p in params]):
        return orig_text

    # just return original text if there is some error with the function

    out = try_except(fn, always(None))(*params)

    # the api calling function can also arrest the process, by returning None

    if not exists(out):
        return orig_text

    # return original text with the output delimiter and the stringified output

    return f'{text_without_end_api_token} {delimiter} {str(out)} {end_api_token}'

# main function, which takes a registry of functions, the text in question, and makes all the appropriate api calls and append the output

def create_function_regex(
    api_start = ' [',
    api_stop = ']'
):
    api_start_regex, api_stop_regex = map(re.escape, (api_start, api_stop))
    return rf'({api_start_regex}(\w+)\(([^)]*)\))({api_stop_regex})'

def num_matches(substr: str, text: str):
    return len(re.findall(re.escape(substr), text))

def has_api_calls(
    text,
    api_start = ' [',
    api_stop = ']'
):
    regex = create_function_regex(api_start, api_stop)
    matches = re.findall(regex, text)
    return len(matches) > 0

def replace_all_but_first(
    text: str,
    api_start = ' [',
    api_stop = ']'
) -> str:
    regex = create_function_regex(api_start, api_stop)

    count = 0

    def replace_(matches):
        orig_text = matches.group(0)
        nonlocal count
        count += 1
        if count > 1:
            return ''
        return orig_text

    return re.sub(regex, replace_, text)

def invoke_tools(
    registry: dict[str, Callable],
    text: str,
    delimiter: str = '→',
    api_start = ' [',
    api_stop = ' ]'
) -> str:
    regex = create_function_regex(api_start, api_stop)
    replace_ = partial(replace_fn, registry, delimiter = delimiter)
    return re.sub(regex, replace_, text)

def invoke_tools_on_batch_sequences(
    registry: dict[str, Callable],
    token_ids: torch.Tensor,
    *,
    encode: Callable,
    decode: Callable,
    delimiter: str = '→',
    api_start = ' [',
    api_stop = ']'
) -> torch.Tensor:
    regex = create_function_regex(api_start_regex, api_stop_regex)
    all_texts = [decode(one_seq_token_ids) for one_seq_token_ids in token_ids]

    invoke_tools_ = partial(invoke_tools, api_start = api_start, api_stop = api_stop)
    all_texts_with_api_calls = [invoke_tools_(registry, text, delimiter) for text in all_texts]

    return encode(all_texts_with_api_calls)

# sampling api related functions
# they do greedy sampling, but encourage sampling api calls by auto-selecting <api> when that token is in the top k = 10

@beartype
@torch.no_grad()
def sample(
    model: nn.Module,
    *,
    seq_len,
    prime: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
    batch_size = 1,
    eos_token_id = None,
    sos_token_id = 1,
    temperature = 0.,
    pad_id = 0,
    call_api_only_once = False,
    api_start_token_id = None,
    auto_select_api_start_token_when_topk = False,
    select_api_start_id_top_k = 10,
):
    device = next(model.parameters()).device
    positions = positions.clone()
    max_seq_len = seq_len + 1

    # validate

    if call_api_only_once:
        assert exists(api_start_token_id)

    # prime

    if exists(prime):
        batch_size, prime_length = prime.shape
    else:
        prime_length = 1
        prime = torch.full((batch_size, 1), sos_token_id, device = device, dtype = torch.long)

    prime = prime.to(device)

    # sampling positions - different sequences have different cursors

    positions = default(positions, torch.zeros((batch_size,), device = device, dtype = torch.long))
    assert (positions <= (prime_length + 1)).all() and (positions <= max_seq_len).all(), 'all positions must be less then initial prime length as well as the total sequence length + 1 (plus one for noop if one sequence finished sampling before the other)'

    # eval model

    model.eval()

    # lengthen the prime to the entire sequence length

    remain_iterations = seq_len - prime_length
    output = F.pad(prime, (0, max_seq_len - prime_length), value = 0.)

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

        # this will ensure that each batch token sequence will have at most one <api> token

        if call_api_only_once:
            if not exists(api_token_mask):
                num_tokens = last_logits.shape[-1]
                api_token_mask = create_api_token_mask(num_tokens, api_start_token_id)
                api_token_mask = api_token_mask.to(device)

            api_called = (output == api_start_token_id).any(dim = -1)

            logit_mask = api_token_mask & rearrange(api_called, 'b -> b 1 1')
            last_logits = last_logits.masked_fill(logit_mask, -torch.finfo(last_logits.dtype).max)

        # greedy sample (but could be made non-greedy)

        sampled = gumbel_sample(last_logits, temperature = temperature)

        # for those sequences without an api call, if the api_start_token_id is within top k (set to 10 in paper) of logits, just auto-select

        # seems to be an important hack in the paper
        # it seems like this paper will take a lot more follow up research to be viable

        if auto_select_api_start_token_when_topk:
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

@beartype
@torch.no_grad()
def sample_with_api_call(
    model: nn.Module,
    *,
    seq_len,
    call_apis: Callable,
    prime: torch.Tensor,
    api_end_token_id: int,
    occurrence = 1,
    **kwargs
):
    sampled = sample(
        model = model,
        prime = prime,
        seq_len = seq_len,
        **kwargs
    )

    sampled = call_apis(sampled)

    sampled_seq_len = sampled.shape[-1]
    null_positions = sampled_seq_len # handle sequences that do not have api calls

    pos_starting_at_end_of_api = find_indices_of(
        sampled,
        api_end_token_id,
        occurrence = occurrence
    )

    resample_after_api_calls = sample(
        model = model,
        prime = sampled,
        seq_len = sampled_seq_len,
        positions = (pos_starting_at_end_of_api + 1).clamp(max = null_positions), # start at the position right after the </api>
        **kwargs
    )

    return resample_after_api_calls

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

    # auto set devices

    device = next(model.parameters()).device
    tokens, tokens_without_api_response, tokens_with_api_response = map(lambda t: t.to(device), (tokens, tokens_without_api_response, tokens_with_api_response))

    # get all the logits

    with torch.no_grad():
        model.eval()
        logits, logits_without_api_response, logits_with_api_response = map(model, (tokens, tokens_without_api_response, tokens_with_api_response))

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
    # this would also assume that the language model perfectly copied the passage over and that both token ids are aligned except for the inserted API call - but this can be done with the custom filtering functions eventually

    weight = weight_and_mask_fn(tokens_without_api_response[:, 1:], api_start_token_id) # shift to the left by one since <api> does not exist in the original sequence
    weight = weight[:, :probs.shape[-1]]

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

# datasets and dataloaders

# for bootstrapping the initial datasets with api calls
# as well as for the final finetuning

@beartype
class PromptDataset(Dataset):
    def __init__(
        self,
        prompt: str,
        prompt_input_tag: str,
        data: List[str],
        tokenizer_encode: Callable
    ):
        self.data = data
        self.prompt = prompt
        self.prompt_input_tag_regex = re.escape(prompt_input_tag)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_string = self.data[idx]
        data_with_prompt = re.sub(self.prompt_input_tag_regex, data_string, self.prompt)
        token_ids = tokenizer.encode(data_with_prompt)
        return torch.tensor(token_ids).long(), torch.tensor(len(token_ids)).long()

def prompt_collate_fn(data, padding_value = 0):
    prompts, prompt_lengths = zip(*data)
    prompts = pad_sequence(prompts, batch_first = True, padding_value = padding_value)
    return prompts, torch.stack(prompt_lengths)

def PromptDataloader(ds: Dataset, *args, padding_value = 0, **kwargs):
    collate_fn = partial(prompt_collate_fn, padding_value = padding_value)
    return DataLoader(ds, *args, collate_fn = collate_fn, **kwargs)

# classes

@beartype
class Toolformer(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        tool_id: str,
        tool: Callable,
        api_start_str = ' [',
        api_stop_str = ']',
        api_response_delimiter = '→',
        api_start_id = None,
        api_stop_id = None,
        teach_tool_prompt: str,
        filter_threshold = 1.,
        pad_id = 0,
        prompt_batch_size = 4,
        model_seq_len = 2048,
        tokenizer_encode: Callable = tokenizer.encode,
        tokenizer_decode: Callable = tokenizer.decode,
        prompt_input_tag: str = DEFAULT_PROMPT_INPUT_TAG,
        exclude_filters: dict[str, Callable[[str], bool]] = dict()
    ):
        super().__init__()
        self.model = model
        self.model_seq_len = model_seq_len

        self.teach_tool_prompt = teach_tool_prompt
        self.prompt_batch_size = prompt_batch_size
        self.prompt_input_tag = prompt_input_tag

        self.tokenizer_encode = tokenizer_encode
        self.tokenizer_decode = tokenizer_decode
        self.tokenizer_encode_to_tensor = lambda s: torch.tensor(tokenizer_encode(s)).long()

        self.filter_threshold = filter_threshold

        self.api_start_str = api_start_str
        self.api_stop_str = api_stop_str
        self.api_response_delimiter = api_response_delimiter

        if not exists(api_start_id):
            api_start_id = tokenizer_encode(api_start_str)
            assert len(api_start_id) == 1
            api_start_id = api_start_id[0]

        self.api_start_id = api_start_id

        if not exists(api_stop_id):
            api_stop_id = tokenizer_encode(api_stop_str)
            assert len(api_stop_id) == 1
            api_stop_id = api_stop_id[0]

        self.api_stop_id = api_stop_id

        self.pad_id = pad_id

        self.tool_id = tool_id
        self.tool = tool
        self.registry = {tool_id: tool}

        assert num_matches(prompt_input_tag, teach_tool_prompt) == 1, f'there must be exactly one prompt input tag `{prompt_input_tag}` in your prompt to encourage the language model to use the designated tool'

        self.teach_tool_prompt = teach_tool_prompt
        self.exclude_filters = exclude_filters

    def generate_data_with_api_calls(
        self,
        data: List[str],
        temperature: float = 0.9
    ) -> List[str]:

        dataset = PromptDataset(
            data = data,
            prompt_input_tag = self.prompt_input_tag,
            prompt = self.teach_tool_prompt,
            tokenizer_encode = self.tokenizer_encode
        )

        dl = PromptDataloader(
            dataset,
            batch_size = self.prompt_batch_size
        )

        prompted_outputs = []

        for prime, positions in dl:

            sampled_outputs = sample(
                model = self.model,
                prime = prime,
                positions = positions,
                seq_len = self.model_seq_len,
                pad_id = self.pad_id,
                temperature = temperature
            )

            for sample_output, position in zip(sampled_outputs, positions):
                start_position = position.item()

                prompted_output = self.tokenizer_decode(sample_output[start_position:])
                prompted_outputs.append(prompted_output)

        return prompted_outputs

    def filter_and_keep_only_first_api_call(
        self,
        data,
        data_with_api_calls: List[str],
        return_excluded = False
    ):
        included = []
        excluded = []

        api_start_stop_kwargs = dict(api_start = self.api_start_str, api_stop = self.api_stop_str)

        has_api_calls_ = partial(has_api_calls, **api_start_stop_kwargs)
        replace_all_but_first_ = partial(replace_all_but_first, **api_start_stop_kwargs)

        for datum, data_with_api_call in zip(data, data_with_api_calls):
            if has_api_calls_(data_with_api_call):
                data_with_api_call = replace_all_but_first_(data_with_api_call)
                included.append((datum, data_with_api_call))
            else:
                excluded.append((datum, data_with_api_call))

        included = tuple(map(list, zip(*included)))

        if not return_excluded:
            return included

        excluded = tuple(map(list, zip(*excluded)))
        return included, excluded

    def make_api_calls(
        self,
        filtered_data_with_api_calls: List[str]
    ):
        invoke_tools_ = partial(
            invoke_tools,
            self.registry,
            api_start = self.api_start_str,
            api_stop = self.api_stop_str, delimiter = self.api_response_delimiter
        )

        data_with_api_responses = []
        for data in filtered_data_with_api_calls:
            output = invoke_tools_(data)
            data_with_api_responses.append(output)

        return data_with_api_responses

    def filter_by_api_responses(
        self,
        data: List[str],
        data_with_api_calls: List[str],
        data_with_api_responses: List[str]
    ) -> FilteredResults:

        to_token_ids = lambda l: pad_sequence([*map(self.tokenizer_encode_to_tensor, l)], batch_first = True, padding_value = self.pad_id)

        tokens, tokens_without_api_response, tokens_with_api_response = map(to_token_ids, (data, data_with_api_calls, data_with_api_responses))

        filtered_results = filter_tokens_with_api_response(
            model = self.model,
            tokens = tokens,
            tokens_with_api_response = tokens_with_api_response,
            tokens_without_api_response = tokens_without_api_response,
            filter_threshold = self.filter_threshold,
            api_start_token_id = self.api_start_id,
            api_end_token_id = self.api_stop_id
        )

        return filtered_results

    def forward(
        self,
        data: List[str]
    ):
        data_with_api_calls = self.generate_data_with_api_calls(data)
        filtered_data_with_api_calls = self.filter_and_keep_only_first_api_call(data_with_api_calls)

        assert len(filtered_data_with_api_calls) > 0, 'your model failed to follow instructions and make API calls. please try a better model or do some better prompt engineering'

        data_with_responses = self.make_api_calls(filtered_data_with_api_calls)

        return data_with_responses
