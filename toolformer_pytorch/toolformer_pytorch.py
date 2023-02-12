import torch
from torch import nn, einsum

from einops import rearrange

# helpers

def exists(val):
    return val is not None

# all we need

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        dim_head = 64,
        heads = 8,
        dropout = 0.
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None
    ):
        b, n, _, device = *x.shape, x.device

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        q = q * self.scale

        # similarities

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        if exists(mask):
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = x.device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # attention

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        # aggregate

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# classes

class Toolformer(nn.Module):
    def __init__(
        self
    ):
        super().__init__()

    def forward(self, x):
        return x
