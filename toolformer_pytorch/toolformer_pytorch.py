import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange

# helpers

def exists(val):
    return val is not None


# normalization
# Layernorm without bias


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# rotary positional embedding
# https://arxiv.org/abs/2104.09864


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, max_seq_len, *, device):
        seq = torch.arange(max_seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = einsum("i , j -> i j", seq, self.inv_freq)
        return torch.cat((freqs, freqs), dim=-1)


def rotate_half(x):
    x = rearrange(x, "... (j d) -> ... j d", j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(pos, t):
    return (t * pos.cos()) + (rotate_half(t) * pos.sin())


# FeedFoward


class FeedForward(nn.Module):
    def __init__(
        self, 
        dim, 
        ff_mult=4, 
        dropout=0.
    ):
        super().__init__()
        ff_inner_dim = int(dim * ff_mult)

        self.ff_out = nn.Sequential(
            nn.Linear(dim, ff_inner_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_inner_dim, dim),
        )

    def forward(self, x):
        return self.ff_out(x)


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
        self.rotary_emb = RotaryEmbedding(dim_head)
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        self.attn_dropout = nn.Dropout(dropout)

        self.register_buffer("pos_emb", None, persistent=False)

    def get_rotary_embedding(self, n, device):
        if self.pos_emb is not None and self.pos_emb.shape[-2] >= n:
            return self.pos_emb[:n]

        pos_emb = self.rotary_emb(n, device=device)
        self.register_buffer("pos_emb", pos_emb, persistent=False)
        return pos_emb

    def forward(
        self,
        x,
        mask = None
    ):
        b, n, _, device, h = *x.shape, x.device, self.heads

        # prenorm

        x = self.norm(x)

        # project for queries, keys, values

        q, k, v = self.to_q(x), *self.to_kv(x).chunk(2, dim = -1)

        # split for multi-headed attention

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        # rotary embeddings

        positions = self.get_rotary_embedding(n, device)
        q, k = map(lambda t: apply_rotary_pos_emb(positions, t), (q, k))

        # scale

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


# GPTJ Block


class GPTJBlock(nn.Module):
    def __init__(
        self, 
        dim, 
        dim_head=64, 
        heads=8, 
        ff_mult=4, 
        dropout=0.
    ):
        super().__init__()
        self.attn = Attention(dim, dim_head=dim_head, heads=heads)
        self.ffn = FeedForward(dim, ff_mult=ff_mult, dropout=dropout)

    def forward(self, x):
        return self.ffn(x) + self.attn(x)


# Transformer


class Transformer(nn.Module):
    def __init__(
        self, 
        dim, 
        depth, 
        heads, 
        dim_head, 
        ff_mult=4,
        dropout=0., 
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(
                GPTJBlock(dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult, dropout=dropout), 
            )

    def forward(self, x):
        for block in self.layers:
            x = block(x) + x
        return x


# classes

class Toolformer(nn.Module):
    def __init__(
        self, 
        dim, 
        num_tokens, 
        depth, 
        dim_head=64, 
        heads=8, 
        ff_mult=4,
        dropout=0.
    ):
        super().__init__()

        self.emb = nn.Embedding(num_tokens, dim)
        self.transformer = Transformer(dim, depth, heads, dim_head, ff_mult, dropout)
        self.to_logits = nn.Linear(dim, num_tokens)

    def forward(self, x):
        x = self.emb(x)
        x = self.transformer(x)
        x = self.to_logits(x)
        return x

if __name__ == "__main__":
    toolformer = Toolformer(
        num_tokens = 20000,
        dim = 512,
        depth = 6,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        dropout = 0.
    )
    tokens = torch.randint(0, 20000, (1, 512))
    logits = toolformer(tokens)
    print(logits.shape)