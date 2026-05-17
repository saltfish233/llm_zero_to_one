import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
import math


class GroupQueryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_kv_head = config.n_kv_head
        self.n_q_head = config.n_q_head
        assert self.n_q_head % self.n_kv_head == 0
        self.head_size = config.head_size
        hidden_dim = config.hidden_dim  # hidden_dim = self.n_q_head * self.head_size
        assert hidden_dim == self.n_q_head * self.head_size
        qkv_dim = (self.n_q_head + 2 * self.n_kv_head) * self.head_size

        self.q_dim = self.n_q_head * self.head_size
        self.kv_dim = self.n_kv_head * self.head_size

        block_size = config.block_size

        self.qkv = nn.Linear(hidden_dim, qkv_dim)
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((block_size, block_size)))
        )
        self.proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        # x (batch_size, seq_len, hidden_dim)
        bs, sl, _ = x.size()

        qkv = self.qkv(x)
        q = (
            qkv[:, :, : self.q_dim]
            .view(bs, sl, self.n_q_head, self.head_size)
            .transpose(1, 2)
        )  # bs,n_q_head,sl,head_size
        k = (
            qkv[:, :, self.q_dim : self.q_dim + self.kv_dim]
            .view(bs, sl, self.n_kv_head, self.head_size)
            .transpose(1, 2)
        )
        v = (
            qkv[:, :, self.q_dim + self.kv_dim : self.q_dim + 2 * self.kv_dim]
            .view(bs, sl, self.n_kv_head, self.head_size)
            .transpose(1, 2)
        )

        k = k.repeat_interleave(self.n_q_head // self.n_kv_head, dim=1)
        v = v.repeat_interleave(self.n_q_head // self.n_kv_head, dim=1)
        # 先不考虑rope
        print(q.size())
        weights = q @ k.transpose(2, 3) / math.sqrt(self.head_size)
        weights = weights.masked_fill(self.attn_mask[:sl, :sl] == 0, float("-inf"))
        attn = F.softmax(weights, dim=-1)  # out -> bs, n_q_head, sl, sl
        out = attn @ v
        out = (
            out.transpose(1, 2).contiguous().reshape(bs, sl, -1)
        )  # out -> bs, sl, hidden_dim
        out = self.proj(out)
        return out


@dataclass
class Config:
    n_q_head: int = 8
    n_kv_head: int = n_q_head // 2
    hidden_dim: int = 768
    head_size: int = hidden_dim // n_q_head
    block_size: int = 512


def main():
    config = Config()
    gqa = GroupQueryAttention(config)

    bs = 16
    sl = 24
    x = torch.rand((bs, sl, config.n_q_head * config.head_size))
    out = gqa(x)
    print(out.shape)


if __name__ == "__main__":

    main()
