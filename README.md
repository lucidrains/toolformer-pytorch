<img src="./toolformer.png" width="500px"></img>

## Toolformer - Pytorch (wip)

Implementation of <a href="https://arxiv.org/abs/2302.04761">Toolformer</a>, Language Models That Can Use Tools, by MetaAI

## Install

```bash
$ pip install toolformer-pytorch
```

## Usage

```python
import torch

from toolformer_pytorch import (
    Toolformer,
    PaLM,
    filter_tokens_with_api_response
)

# model

palm = PaLM(
    dim = 512,
    num_tokens = 20000,
    depth = 2,
    heads = 8,
    dim_head = 64
).cuda()

# mock some tokens

mock_start_pos = 512
mock_api_call_length = 10
mock_api_start_id = 19998
mock_api_stop_id = 19999

tokens = torch.randint(0, 20000, (10, 1024)).cuda()
tokens_with_api_response = torch.randint(0, 20000, (10, 1024)).cuda()
tokens_without_api_response = torch.randint(0, 20000, (10, 1024)).cuda()

tokens_with_api_response[:, mock_start_pos] = mock_api_start_id
tokens_with_api_response[:, mock_start_pos + mock_api_call_length] = mock_api_stop_id

tokens_without_api_response[:, mock_start_pos] = mock_api_start_id
tokens_without_api_response[:, mock_start_pos + mock_api_call_length] = mock_api_stop_id

# filter

filtered_results = filter_tokens_with_api_response(
    model = palm,
    tokens = tokens,
    tokens_with_api_response = tokens_with_api_response,
    tokens_without_api_response = tokens_without_api_response,
    filter_threshold = 1.,
    api_start_token_id = mock_api_start_id,
    api_end_token_id = mock_api_stop_id
)
```

## Todo

- [ ] create custom generate function for palm that can do external API calls
- [ ] do end-to-end training in `Toolformer`
- [ ] hook up gpt-j
- [ ] test for a simple calculator eval dataset

## Citations

```bibtex
@inproceedings{Schick2023ToolformerLM,
    title   = {Toolformer: Language Models Can Teach Themselves to Use Tools},
    author  = {Timo Schick and Jane Dwivedi-Yu and Roberto Dessi and Roberta Raileanu and Maria Lomeli and Luke Zettlemoyer and Nicola Cancedda and Thomas Scialom},
    year    = {2023}
}
```

```bibtex
@article{Gao2022PALPL,
    title   = {PAL: Program-aided Language Models},
    author  = {Luyu Gao and Aman Madaan and Shuyan Zhou and Uri Alon and Pengfei Liu and Yiming Yang and Jamie Callan and Graham Neubig},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2211.10435}
}
```
