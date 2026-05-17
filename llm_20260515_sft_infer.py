"""
20260511: 加入更新技术: RoPE、GQA
20260513: 加入RMSNorm;
20260514: 采用SwiGLU; 改造rope为预先计算freqs; 加入tie weight
20260516: 加入SFT逻辑; 注意模型loss计算时在模型里去错位，而不是在dataset中
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
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from tqdm import tqdm
from datasets import load_dataset, Features, Value
import random
from transformers import AutoTokenizer


@dataclass
class GPTConfig:
    hidden_size: int = 768
    n_heads: int = 8
    n_kv_heads: int = n_heads // 2
    n_layers: int = 8
    head_size: int = hidden_size // n_heads
    vocab_size: int = 6400
    block_size: int = 1024
    dropout: float = 0.1
    intermediate_size: int = math.ceil(hidden_size * math.pi / 64) * 64
    batch_size: int = 32
    device: str = "cuda:0"
    num_epochs: int = 2
    train_bin: str = (
        "/data/lqc/Practice/llm/from_zero_to_one/corpus/pretrain_t2t_minimind.bin"
    )
    sft_data_path: str = "/data/lqc/Practice/llm/from_zero_to_one/corpus/sft_t2t.jsonl"
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
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size))),persistent=False
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
            logits = logits[:,:-1,:]
            targets = targets[:,1:]
            out = logits.reshape(bs * (sl-1), -1)
            targets = targets.reshape(bs * (sl-1))
            loss = F.cross_entropy(out, target=targets)
        else:
            loss = None
        return logits, loss

    @torch.no_grad()  # 优化 1：推理时绝对不能计算梯度，否则显存会爆炸！
    def generate(self, ids, max_new_tokens, temperature=0.8, top_k=50, top_p=0.85, repetition_penalty=1.1, do_sample=True, eos_token_id=2):
        self.eval()  # 确保模型在评估模式（关闭 Dropout 等）
        
        for i in range(max_new_tokens):
            # 截断 context window 到 block_size
            context_ids = ids if ids.size(1) <= self.block_size else ids[:, -self.block_size :]

            # 获取 logits
            logits, _ = self(context_ids)
            next_token_logits = logits[:, -1, :]  # 只取最后一个预测的 token 的 logits

            # 优化 2：重复惩罚 (Repetition Penalty)
            # 压低已经生成过的 token 的概率，防止模型变成“复读机”
            if repetition_penalty != 1.0:
                for b in range(ids.size(0)):
                    seen_tokens = torch.unique(ids[b])
                    score = next_token_logits[b, seen_tokens]
                    # 如果 logit 大于 0，除以惩罚值；小于 0，乘以惩罚值（使其变得更小）
                    next_token_logits[b, seen_tokens] = torch.where(
                        score > 0, score / repetition_penalty, score * repetition_penalty
                    )

            # 优化 3：温度缩放 (Temperature)
            # 温度越低，模型越确定；温度越高，模型越发散
            if temperature > 0:
                next_token_logits = next_token_logits / temperature

            if do_sample and temperature > 0:
                # 优化 4：Top-K 截断
                # 只保留概率最高的 K 个词，直接一刀切掉长尾的“离谱词”
                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    min_top_k = top_k_values[..., -1, None]
                    next_token_logits = torch.where(
                        next_token_logits < min_top_k,
                        torch.full_like(next_token_logits, float('-inf')),
                        next_token_logits
                    )

                # 优化 5：Top-P 截断 (Nucleus Sampling)
                # 按概率从大到小累加，超过 top_p 阈值后的词全部扔掉
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 找出需要被移除的 token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 把掩码向右平移一位，确保至少保留概率最大的那一个词
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # 将需要移除的词的 logits 设为负无穷
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')

                # 采样！经过层层过滤，现在的概率分布已经很安全了
                probs = F.softmax(next_token_logits, dim=-1)
                preds = torch.multinomial(probs, num_samples=1)
            else:
                # 优化 6：贪心搜索支持
                # 如果 do_sample=False，直接选择概率最大的词，适合回答数学、代码等确定性问题
                preds = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            if preds.item() == eos_token_id:  # 遇到结束符就停止
                break
                
            ids = torch.cat((ids, preds), dim=1)
            
        return ids

def pre_process_chat(conversation, add_sys_ratio=0.2):
    if any(conv.get("tools") for conv in conversation): return conversation

    SYSTEM_PROMPT = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是KuriyamaMirai，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是KuriyamaMirai，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are KuriyamaMirai, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are KuriyamaMirai, a small but useful language model."
    ]
    
    if conversation[0].get("role") != "system" and random.random() < add_sys_ratio:
        conversation = [{"role": "system", "content": random.choice(SYSTEM_PROMPT)}] + conversation
    return conversation

def post_process_chat(prompt_content, ratio=0.2):
    pattern = "<think>\n\n</think>\n\n"
    if pattern in prompt_content and random.random() > ratio:
        prompt_content = prompt_content.replace(pattern, "")
    return prompt_content


class MySFTDataset(Dataset):
    def __init__(self, config, tokenizer, file_path=None):
        super().__init__()
        self.block_size = config.block_size
        self.tokenizer = tokenizer
        features = Features({
            "conversations": [{
                "role": Value("string"),
                "content": Value("string"),
                "reasoning_content": Value("string"),
                "tools": Value("string"),
                "tool_calls": Value("string")
            }]
        })

        self.conversations = load_dataset("json", data_files=file_path, split="train", features=features)

        self.bos_id = self.tokenizer(f"{tokenizer.bos_token}assistant\n", add_special_tokens=False).input_ids
        self.eos_id = self.tokenizer(f"{tokenizer.eos_token}\n", add_special_tokens=False).input_ids
    
    def generate_labels(self, input_ids):
        labels = [-100] * len(input_ids)
        start = 0
        max_len = min(self.block_size, len(input_ids))  - len(self.bos_id)
        while start < max_len:
            if input_ids[start: start+len(self.bos_id)] == self.bos_id:
                end = start+len(self.bos_id)
                if end >= self.block_size: break
                found_eos = False
                for j in range(end, min(self.block_size, len(input_ids))):
                    if input_ids[j:j+len(self.eos_id)] == self.eos_id:
                        found_eos = True
                        labels[j:j+len(self.eos_id)] = input_ids[j:j+len(self.eos_id)]
                        start = j + len(self.eos_id)
                        break
                    labels[j] = input_ids[j]
                if found_eos is not True:
                    start = max_len
            else:
                start += 1
        return labels


    def create_chat_prompt(self, conversation):
        messages = []
        tools = None
        for message in conversation:
            if message.get("role") == "system" and message.get("tools"):
                tools = json.loads(message["tools"]) if isinstance(message["tools"], str) else message["tools"]
            if message.get("tool_calls"):
                message["tool_calls"] = json.loads(message["tool_calls"])

            messages.append(message)
        
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
            tools=tools
        )

    def __getitem__(self, idx):
        conversation = self.conversations[idx]["conversations"]
        conversation = pre_process_chat(conversation=conversation)
        prompt = self.create_chat_prompt(conversation=conversation)
        prompt = post_process_chat(prompt_content=prompt)
        input_ids = self.tokenizer(prompt).input_ids[:self.block_size]
        input_ids = input_ids + [self.tokenizer.pad_token_id] * (self.block_size - len(input_ids))
        labels = self.generate_labels(input_ids=input_ids)
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.conversations)



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
        "/data/lqc/Practice/llm/from_zero_to_one/checkpoints/20260517_sft_epoch_0_train-loss_1.4059",
        map_location=GPTConfig.device,
    )
    model_weights = state_dict["model_state_dict"]
    unwrapped_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
    model.load_state_dict(unwrapped_weights)

    # 计算模型总参数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total_params / 1e6:.2f}M")

    model.eval()
    prompt = [
        {"role": "system", "content": "你是一个知识丰富的AI，尽力为用户提供准确的信息。"},
        {"role": "user", "content": "写一首关于自由的诗？"},
    ]
    enc = AutoTokenizer.from_pretrained(GPTConfig.tokenizer_path)
    prompt = enc.apply_chat_template(prompt, tokenize=True,add_generation_prompt=True, open_thinking=True)
    # print(prompt)
    prompt = prompt["input_ids"]
    input_ids = torch.tensor(prompt).unsqueeze(0).to(GPTConfig.device)
    res_tokens = model.generate(input_ids, max_new_tokens=1024)[0].tolist()
    res_text = enc.decode(res_tokens, skip_special_tokens=True)
    print(res_text)