import torch
from torch import nn
from dataclasses import dataclass


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate = config.intermediate

        self.gate_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate, bias=False)
        self.down_proj = nn.Linear(intermediate, hidden_size, bias=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))


@dataclass
class Config:
    hidden_size: int = 768
    intermediate: int = 4 * hidden_size


def main():
    conf = Config()
    x = torch.rand((16, 24, 768))
    ffn = FeedForward(conf)
    out = ffn(x)
    print(out.shape)

main()
