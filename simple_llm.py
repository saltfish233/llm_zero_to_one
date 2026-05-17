import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import math
from dataclasses import dataclass
import tiktoken
import json


@dataclass
class GPTConfig:
    batch_size: int = 8
    block_size: int = 512
    n_embed: int = 1024
    n_heads: int = 8
    n_layers: int = 12
    hidden_dim: int = n_embed
    head_size: int = int(hidden_dim / n_heads)

    dropout: float = 0.1
    vocab_size: int = 50257
    num_epochs: int = 16


# single-head att / multi-head att / FeedForward
class SingleHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.hidden_dim
        self.head_size = config.head_size

        self.query = nn.Linear(hidden_dim, self.head_size)
        self.key = nn.Linear(hidden_dim, self.head_size)
        self.value = nn.Linear(hidden_dim, self.head_size)

        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer(
            "attn_mask", torch.tril(torch.ones((config.block_size, config.block_size)))
        )

    def forward(self, x):
        # x shape -> (b,s,d)
        batch_size, seq_len, emb_dim = x.size()
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        weight = q @ k.transpose(-1, -2) / math.sqrt(self.head_size)
        weight = weight.masked_fill(
            self.attn_mask[:seq_len, :seq_len] == 0, float("-inf")
        )
        weight = self.dropout(F.softmax(weight, dim=-1))
        out = weight @ v

        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_heads = config.n_heads
        hidden_dim = config.hidden_dim

        self.heads = nn.ModuleList(
            [SingleHeadSelfAttention(config) for _ in range(n_heads)]
        )
        self.projector = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.projector(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.hidden_dim, 4 * config.hidden_dim),
            nn.GELU(),
            nn.Linear(4 * config.hidden_dim, config.hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.msa = MultiHeadSelfAttention(config)
        self.ffn = FeedForward(config)

        self.ln1 = nn.LayerNorm(config.hidden_dim)
        self.ln2 = nn.LayerNorm(config.hidden_dim)

    def forward(self, x):
        x = self.msa(self.ln1(x)) + x
        x = self.ffn(self.ln2(x)) + x
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.block_size, config.n_embed)

        self.layers = nn.Sequential(*[Block(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.hidden_dim)
        self.unembedding = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        self.token_embedding_table.weight = self.unembedding.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, ids, targets=None):
        batch_size, seq_len = ids.size()
        token_embedding = self.token_embedding_table(
            ids
        )  # token_embedding -> bs,sl,n_embed
        position_embedding = self.position_embedding_table(
            torch.arange(0, seq_len, device=ids.device)
        )

        # print(1)
        x = token_embedding + position_embedding
        # print(2)
        x = self.layers(x)
        x = self.ln_final(x)
        logits = self.unembedding(x)  # logits -> bs,sl,vocab_size

        if targets is None:
            loss = None
        else:
            _, _, vocab_size = logits.size()
            logits = logits.view(batch_size * seq_len, vocab_size)
            targets = targets.view(batch_size * seq_len)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generator(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = (
                idx if idx.size(1) <= max_new_tokens else idx[:, -max_new_tokens:]
            )
            logits, _ = self(idx_cond)
            probs = torch.softmax(logits[:, -1, :], dim=-1)
            pred = torch.multinomial(probs, 1)
            idx = torch.cat((idx, pred), dim=1)
        return idx


class MyDataset(Dataset):
    def __init__(self, config):
        import tiktoken

        self.enc = tiktoken.get_encoding("gpt2")
        block_size = config.block_size

        self.eos_token = self.enc.encode(
            "<|endoftext|>", allowed_special={"<|endoftext|>"}
        )

        file_path = "/data/lqc/Practice/llm/from_zero_to_one/corpus/seq-monkey-data/chinese_general/mobvoi_seq_monkey_general_open_corpus.jsonl"

        raw_data = []
        max_len = 2000
        with open(file_path) as f:
            for i, line in enumerate(f):
                if i >= max_len:
                    break
                try:
                    data = json.loads(line.strip())
                    raw_data.append(data["text"])
                except Exception as e:
                    # print(e)
                    continue

        token_extends = []
        for data in raw_data:
            # print(data.type)
            data_token = self.encode(data)
            tokens = data_token + self.eos_token
            token_extends.extend(tokens)

        # 长 -> 短
        self.encoded_tokens = []
        for i in range(0, len(token_extends), block_size):
            chunk = token_extends[i : i + block_size + 1]
            # print("chunk", len(chunk))
            if len(chunk) < block_size + 1:
                chunk = chunk + self.eos_token * (block_size + 1 - len(chunk))
            self.encoded_tokens.append(chunk)

    def __getitem__(self, idx):
        chunk = self.encoded_tokens[idx]
        # print(1)
        # print(type(chunk))
        try:
            ids = torch.tensor(chunk[:-1], dtype=torch.long)
        except Exception as e:
            print(chunk[:-1])
        # print(2)
        targets = torch.tensor(chunk[1:], dtype=torch.long)
        # print(3)
        return ids, targets

    def __len__(self):
        return len(self.encoded_tokens)

    def encode(self, text):
        return self.enc.encode(text)


# ds = MyDataset(GPTConfig)
# train_ds, val_ds = torch.utils.data.random_split(ds, [0.9, 0.1])
# train_dl = DataLoader(train_ds, GPTConfig.batch_size, shuffle=True)
# val_dl = DataLoader(val_ds, GPTConfig.batch_size)


device = "cuda:0"

model = GPT(GPTConfig).to(device)

total_params = sum(p.numel() for p in model.parameters())
print(total_params / 1e6, "M")

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer=optimizer, T_max=1000
)


def train(model, optimizer, scheduler, train_loader, device, epoch):
    model.train()
    train_loss = 0
    for batch_ids, (x, y) in enumerate(train_loader):
        x, y = x.to(device), y.to(device)
        _, loss = model(x, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss.item()

        if batch_ids % 100 == 0:
            print(
                f"epoch {epoch} batch {batch_ids} train loss {train_loss / (batch_ids+1):.4f}"
            )

    return train_loss / len(train_loader)


def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            val_loss += loss.item()

    return val_loss / len(val_loader)


# for epoch in range(GPTConfig.num_epochs):
#     train_loss = train(model, optimizer, lr_scheduler, train_dl, device, epoch)

#     val_loss = eval(model, val_dl, device)
#     print(f"epoch {epoch} avg train loss {train_loss:.4f} avg val loss {val_loss:.4f}")

#     checkpoint = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "scheduler_state_dict": lr_scheduler.state_dict(),
#         "val_loss": val_loss,
#     }

#     torch.save(checkpoint, f"checkpoints/epoch_{epoch}.pth")

if __name__ == "__main__":
    device = "cuda:0"
    checkpoint = torch.load(
        "/data/lqc/Practice/llm/from_zero_to_one/checkpoints/epoch_15.pth",
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    enc = tiktoken.get_encoding("gpt2")

    eos = enc.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
    prompt = "你好"
    prompt_tokens = eos + enc.encode(prompt)
    prompt_tokens = torch.tensor(
        prompt_tokens, dtype=torch.long, device=device
    ).unsqueeze(0)

    # print(prompt_tokens)
    tokens = model.generator(prompt_tokens, 100)
    tokens = tokens[0].tolist()
    res = enc.decode(tokens)
    print(res)

    # model.generator()
