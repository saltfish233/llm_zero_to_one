"""
20260511: 加入更新技术: RoPE、GQA
20260513: 加入RMSNorm;
20260514: 采用SwiGLU; 改造rope为预先计算freqs; 加入tie weight
"""

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

os.environ["HF_DATASETS_CACHE"] = "/data/lqc/Practice/llm/from_zero_to_one/cache"
from tqdm import tqdm
from transformers import AutoTokenizer


@dataclass
class GPTConfig:
    hidden_size: int = 768
    n_heads: int = 8
    n_kv_heads: int = n_heads // 2
    n_layers: int = 8
    head_size: int = hidden_size // n_heads
    vocab_size: int = 6400
    block_size: int = 512
    dropout: float = 0.1
    intermediate_size: int = math.ceil(hidden_size * math.pi / 64) * 64
    batch_size: int = 114
    device: str = "cuda:0"
    num_epochs: int = 2
    train_bin: str = (
        "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_minimind.bin"
    )
    data_path: str = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t.jsonl"
    tokenizer_path: str = "/data/lqc/Practice/llm/from_zero_to_one/tokenizers/minimind"
    eps: float = 1e-5


def precompute_freqs_cis(block_size, head_size, rope_theta: float = 1e6, device="cpu"):
    assert head_size % 2 == 0, "head_size must be even for RoPE"
    theta = rope_theta ** -(2 * torch.arange(head_size // 2, device=device) / head_size)
    pos = torch.arange(block_size, device=device)
    freqs = torch.outer(pos, theta)
    freq_cos = freqs.cos().repeat_interleave(2, dim=-1)
    freq_sin = freqs.sin().repeat_interleave(2, dim=-1)
    return freq_cos, freq_sin


def rotary_embedding(q: torch.Tensor, k: torch.Tensor, freq_cos, freq_sin):

    batch_size, n_heads, seq_len, head_size = q.size()

    q_rotate = torch.stack([-q[:, :, :, 1::2], q[:, :, :, ::2]], dim=-1).reshape(
        batch_size, n_heads, seq_len, head_size
    )
    k_rotate = torch.stack([-k[:, :, :, 1::2], k[:, :, :, ::2]], dim=-1).reshape(
        batch_size, k.size(1), seq_len, head_size
    )

    return (
        q * freq_cos + q_rotate * freq_sin,
        k * freq_cos + k_rotate * freq_sin,
    )


class RMSNorm(nn.Module):
    def __init__(self, config, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = config.eps

    def _norm(self, x: torch.Tensor):
        try:
            return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        except:
            print(x)
            exit(0)

    def forward(self, x: torch.Tensor):
        return self.weight * self._norm(x.float()).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.head_size = config.head_size
        # self.qkv_size = config.head_size * config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.q_size = config.head_size * config.n_heads
        self.kv_size = config.head_size * config.n_kv_heads

        self.qkv = nn.Linear(config.hidden_size, self.q_size + self.kv_size * 2)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.dropout = nn.Dropout(config.dropout)
        self.project = nn.Linear(config.hidden_size, config.hidden_size)

        self.q_norm = RMSNorm(config, config.head_size)
        self.k_norm = RMSNorm(config, config.head_size)

        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size)))
        )

    def forward(self, x, pos_embedding):
        bs, sl, _ = x.size()
        qkv_matrix = self.qkv(x)
        q, k, v = qkv_matrix.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(bs, sl, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(bs, sl, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(bs, sl, self.n_kv_heads, self.head_size).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)

        # 先rotary 再 repeat
        freqs_cos, freqs_sin = pos_embedding
        q, k = rotary_embedding(q, k, freqs_cos, freqs_sin)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        weights = (
            q @ k.transpose(-2, -1) / math.sqrt(self.head_size)
        )  # weights bs,hd,sl,sl
        weights = weights.masked_fill(self.attn_mask[:sl, :sl] == 0, float("-inf"))
        attn = self.attn_dropout(F.softmax(weights.float(), dim=-1).type_as(weights))
        out = attn @ v  # out -> bs, hd , sl, head_size (hd = head_size * n_head)
        out = out.transpose(1, 2).reshape(bs, sl, -1)

        out = self.project(out)
        out = self.dropout(out)
        return out


# swiglu
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.mha = MultiHeadAttention(config)
        self.ln1 = RMSNorm(config, config.hidden_size)
        self.ffn = FeedForward(config)
        self.ln2 = RMSNorm(config, config.hidden_size)

    def forward(self, x, pos_embedding):
        x = x + self.mha(self.ln1(x), pos_embedding)
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block_size = config.block_size
        self.vocab_embedding_table = nn.Embedding(config.vocab_size, config.hidden_size)

        self.net = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln_final = RMSNorm(config, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        freqs_cos, freqs_sin = precompute_freqs_cis(
            self.block_size, config.head_size, device=config.device
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)
        # tie weight
        self.vocab_embedding_table.weight = self.lm_head.weight

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

        embedding = self.vocab_embedding_table(ids)

        pos_embedding = (self.freqs_cos[:sl, :], self.freqs_sin[:sl, :])
        for block in self.net:
            embedding = block(embedding, pos_embedding)
        logits = self.ln_final(embedding)
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
            if preds.item() == 2:  # <|endoftext|>
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

        x = chunk[:-1].clone()
        y = chunk[1:].clone()
        y[x==2] = -100

        return x, y

    def __len__(self):
        return self.total_chunks


def train(model, optimizer, scheduler, train_dl, device, epoch):
    model.train()
    train_loss = 0
    total_batchs = len(train_dl)

    # pbar = tqdm(enumerate(train_dl), total=total_batchs, disable=(local_rank != 0), dynamic_ncols=True)

    for batch_idx, (x, y) in enumerate(train_dl):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            _, loss = model(x, y)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        train_loss += loss.item()

        if local_rank == 0 and (batch_idx + 1) % 200 == 0:
            avg_loss = train_loss / (batch_idx + 1)
            print(
                f"epoch {epoch} step {batch_idx + 1}/{total_batchs} | train_loss {avg_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.6f}"
            )

    return train_loss / total_batchs


def val(model, val_dl, device):
    model.eval()
    val_loss = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for x, y in val_dl:
            x, y = x.to(device), y.to(device)
            with autocast_ctx:
                _, loss = model(x, y)
            val_loss += loss

    avg_loss = val_loss / len(val_dl)
    if dist.is_initialized():
        dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
        avg_loss = avg_loss / dist.get_world_size()

    return avg_loss.item()


def init_distributed_mode():
    if int(os.environ.get("RANK", -1)) == -1:
        return 0
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK"))
    torch.cuda.set_device(local_rank)
    return local_rank


# def init_model(config):


if __name__ == "__main__":
    model = GPT(GPTConfig).to(GPTConfig.device)

    state_dict = torch.load(
        "/data/lqc/Practice/llm/from_zero_to_one/checkpoints/20260515_epoch_0_train-loss_2.0357",
        map_location=GPTConfig.device,
    )
    model.load_state_dict(state_dict["model_state_dict"])

    # 计算模型总参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    model.eval()
    prompt = "法国的首都是巴黎"
    enc = AutoTokenizer.from_pretrained(GPTConfig.tokenizer_path)
    ids = torch.tensor(
        [enc.bos_token_id] + enc.encode(prompt),
        device=GPTConfig.device,
    ).unsqueeze(0)
    res_tokens = model.generate(ids, 500)[0].tolist()
    res = enc.decode(res_tokens[1:])
    print(res)
