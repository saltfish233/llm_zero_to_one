"""
加入更新技术: RoPE、GQA
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


@dataclass
class GPTConfig:
    hidden_size: int = 768
    n_heads: int = 8
    n_kv_heads: int = n_heads // 2
    n_layers: int = 12
    head_size: int = hidden_size // n_heads
    vocab_size: int = 50257
    block_size: int = 512
    dropout: float = 0.1
    batch_size: int = 48
    device: str = "cuda:0"
    num_epochs: int = 2
    train_bin: str = (
        "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_train.bin"
    )
    val_bin: str = "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_val.bin"


def rotary_embedding(q: torch.Tensor, k: torch.Tensor):

    device = q.device
    batch_size, n_heads, seq_len, head_size = q.size()
    assert head_size % 2 == 0

    theta = 10000 ** -(2 * torch.arange(head_size // 2, device=device) / head_size)
    pos = torch.arange(seq_len, device=device)
    pos_theta = torch.outer(pos, theta)  # pos_theta.shape -> seq_len,head_size//2
    pos_theta = pos_theta.repeat_interleave(2, dim=-1)

    q_rotate = torch.stack([-q[:, :, :, 1::2], q[:, :, :, ::2]], dim=-1).reshape(
        batch_size, n_heads, seq_len, head_size
    )
    k_rotate = torch.stack([-k[:, :, :, 1::2], k[:, :, :, ::2]], dim=-1).reshape(
        batch_size, k.size(1), seq_len, head_size
    )

    return (
        q * pos_theta.cos() + q_rotate * pos_theta.sin(),
        k * pos_theta.cos() + k_rotate * pos_theta.sin(),
    )


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

        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size)))
        )

    def forward(self, x):
        bs, sl, _ = x.size()
        qkv_matrix = self.qkv(x)
        q, k, v = qkv_matrix.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(bs, sl, self.n_heads, self.head_size).transpose(1, 2)
        k = k.view(bs, sl, self.n_kv_heads, self.head_size).transpose(1, 2)
        v = v.view(bs, sl, self.n_kv_heads, self.head_size).transpose(1, 2)

        # 先rotary 再 repeat
        q, k = rotary_embedding(q, k)
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

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

        embedding = self.vocab_embedding_table(ids)

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


def train(model, optimizer, scheduler, train_dl, device, epoch):
    model.train()
    train_loss = 0
    total_batchs = len(train_dl)
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

        if local_rank == 0 and (batch_idx + 1) % 1000 == 0:
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


if __name__ == "__main__":
    # DDP
    local_rank = init_distributed_mode()
    if dist.is_initialized():
        GPTConfig.device = f"cuda:{local_rank}"

    # Mix precision
    dtype = torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)

    model = GPT(GPTConfig).to(GPTConfig.device)
    if dist.is_initialized():
        model = DistributedDataParallel(model, device_ids=[local_rank])

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3000)

    train_ds = MyDataset(GPTConfig, GPTConfig.train_bin)
    val_ds = MyDataset(GPTConfig, GPTConfig.val_bin)

    train_sampler = DistributedSampler(train_ds)
    train_dl = DataLoader(
        train_ds,
        GPTConfig.batch_size,
        sampler=train_sampler,
        pin_memory=True,
        num_workers=8,
    )

    val_sampler = DistributedSampler(val_ds)
    val_dl = DataLoader(
        val_ds,
        GPTConfig.batch_size,
        sampler=val_sampler,
        num_workers=8,
        pin_memory=True,
    )

    for epoch in range(GPTConfig.num_epochs):
        train_sampler.set_epoch(epoch)

        train_loss = train(
            model, optimizer, scheduler, train_dl, GPTConfig.device, epoch
        )
        val_loss = val(model, val_dl, GPTConfig.device)
        if local_rank == 0:
            print(
                f"Epoch{epoch} avg_train_loss {train_loss:.4f} avg_val_loss {val_loss:.4f}"
            )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if local_rank == 0:
            torch.save(checkpoint, f"checkpoints/epoch_{epoch}_valloss_{val_loss:.4f}")

    # state_dict = torch.load(
    #     "/data/lqc/Practice/llm/from_zero_to_one/checkpoints/epoch_13_valloss_1.8539",
    #     map_location=GPTConfig.device,
    # )
    # model.load_state_dict(state_dict["model_state_dict"])
    # model.eval()
    # prompt = "中国的首都是"
    # enc = tiktoken.get_encoding("gpt2")
    # eos = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    # ids = torch.tensor(eos + enc.encode(prompt), device=GPTConfig.device).unsqueeze(0)
    # res_tokens = model.generate(ids, 400)[0].tolist()
    # res = enc.decode(res_tokens)
    # print(res)
