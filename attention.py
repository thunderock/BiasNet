# @Filename:    attention.py
# @Author:      Ashutosh Tiwari
# @Email:       checkashu@gmail.com
# @Time:        4/9/22 4:12 AM

import torch
from torch import nn
from torch.nn import functional as F
import math
import pytorch_lightning as pl

class MultiheadAttention(nn.Module):

# https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    def __init__(self, input_dim, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        print(input_dim, 3 * embed_dim)
        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None):
        d_k = q.size()[-1]
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, x, mask=None, return_attention=False):

        print("x.shape: ", x.shape )
        batch_size, y, seq_length, embed_dim = x.size()


        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = self.scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.out_proj(values)

        if return_attention:
            return o, attention
        else:
            return o

#
# seq_len, d_k = 3, 2
# pl.seed_everything(42)
# q = torch.randn(seq_len, d_k)
# k = torch.randn(seq_len, d_k)
# v = torch.randn(seq_len, d_k)
# values, attention = MultiheadAttention.scaled_dot_product(q, k, v)
# print("Q\n", q)
# print("K\n", k)
# print("V\n", v)
# print("Values\n", values)
# print("Attention\n", attention)