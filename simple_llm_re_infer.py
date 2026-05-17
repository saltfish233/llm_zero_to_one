import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import math
from dataclasses import dataclass
import tiktoken
import json
import numpy as np
import os


@dataclass
class GPTConfig:
    hidden_size: int = 768
    n_heads: int = 8
    n_layers: int = 12
    head_size: int = hidden_size // n_heads
    vocab_size: int = 50257
    block_size: int = 512
    dropout: float = 0.1
    batch_size: int = 48
    device: str = "cuda:5"
    num_epochs: int = 2
    train_bin: str = (
        "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_train.bin"
    )
    val_bin: str = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_val.bin"


class SingleHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.query = nn.Linear(config.hidden_size, config.head_size)
        self.key = nn.Linear(config.hidden_size, config.head_size)
        self.value = nn.Linear(config.hidden_size, config.head_size)

        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size)))
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # print(x.shape)
        bs, sl, _ = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        # print("q ", q.size())
        weights = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        weights = weights.masked_fill(self.attn_mask[:sl, :sl] == 0, float("-inf"))
        att = self.dropout(F.softmax(weights, dim=-1))
        return att @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_size = config.head_size
        self.qkv_size = config.head_size * config.n_heads

        self.qkv = nn.Linear(config.hidden_size, self.qkv_size * 3)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.project = nn.Linear(config.hidden_size, config.hidden_size)

        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size)))
        )

    def forward(self, x):
        bs, sl, _ = x.size()
        qkv_matrix = self.qkv(x)
        q, k, v = qkv_matrix.split(self.qkv_size, dim=-1)
        q = q.view(bs, sl, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(bs, sl, self.n_heads, self.head_size).transpose(1, 2)
        v = v.view(bs, sl, self.n_heads, self.head_size).transpose(1, 2)

        weights = (
            q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        )  # weights bs,hd,sl,sl
        weights = weights.masked_fill(self.attn_mask[:sl, :sl] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(weights, dim=-1))
        out = attn @ v  # out -> bs, hd , sl, head_size (hd = head_size * n_head)
        out = out.transpose(1, 2).reshape(bs, sl, -1)

        out = self.project(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ffn = FeedForward(config)
        self.ln2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding_table = nn.Embedding(
            config.block_size, config.hidden_size
        )

        self.net = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, ids, targets=None):
        bs, sl = ids.size()
        # print(ids)

        ids_embedding = self.vocab_embedding_table(ids)
        pos_embedding = self.position_embedding_table(
            torch.arange(sl, device=ids.device)
        )

        embedding = ids_embedding + pos_embedding
        logits = self.net(embedding)
        logits = self.ln_final(logits)
        logits = self.lm_head(logits)

        if targets is not None:
            out = logits.view(bs * sl, -1)
            targets = targets.view(bs * sl)
            loss = F.cross_entropy(out, target=targets)
        else:
            loss = None
        return logits, loss

    def generate(self, ids, max_new_tokens):
        for i in range(max_new_tokens):
            ids = ids if ids.size(1) <= self.block_size else ids[:, -self.block_size :]

            logits, _ = self(ids)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            preds = torch.multinomial(probs, 1)
            if preds.item() == 50256:  # <|endoftext|>
                break
            ids = torch.cat((ids, preds), dim=1)
        return ids


class MyDataset(Dataset):
    def __init__(self, config, file_path=None):
        self.block_size = config.block_size

        self.data = np.memmap(file_path, dtype=np.uint16, mode="r")
        self.total_chunks = len(self.data) // (config.block_size + 1)

    def __getitem__(self, idx):
        start_idx = idx * (self.block_size + 1)
        end_idx = start_idx + self.block_size + 1
        chunk = torch.from_numpy(self.data[start_idx:end_idx].astype(np.int64))

        x = chunk[:-1]
        y = chunk[1:]
        # print(x)
        return x, y

    def __len__(self):
        return self.total_chunks


if __name__ == "__main__":
    model = GPT(GPTConfig).to(GPTConfig.device)

    state_dict = torch.load(
        "/data/lqc/Practice/llm/from_zero_to_one/checkpoints/epoch_0_valloss_1.2822",
        map_location=GPTConfig.device,
    )
    model.load_state_dict(state_dict["model_state_dict"])
    model.eval()
    prompt = "北京大学在"
    enc = tiktoken.get_encoding("gpt2")
    eos = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    ids = torch.tensor(eos + enc.encode(prompt), device=GPTConfig.device).unsqueeze(0)
    res_tokens = model.generate(ids, 500)[0].tolist()
    res = enc.decode(res_tokens)
    print(res)
