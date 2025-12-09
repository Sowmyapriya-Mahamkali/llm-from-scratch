import torch
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        B, T, D = x.size()
        x = x.view(B, T, self.num_heads, self.d_head)
        return x.permute(0,2,1,3)

    def combine_heads(self, x):
        x = x.permute(0,2,1,3).contiguous()
        B, T, H, d = x.size()
        return x.view(B, T, H * d)

    def forward(self, x, mask=None):
        B, T, D = x.size()
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)
        scores = (q @ k.transpose(-2,-1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = attn @ v
        out = self.combine_heads(out)
        out = self.out(out)
        return out
