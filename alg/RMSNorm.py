import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        return self.gamma * self._norm(x.float()).type_as(x)


def main():
    bs = 2
    sl = 4
    dim = 128
    x = torch.rand((bs, sl, dim))
    nm = RMSNorm(dim=dim)
    print(nm(x).size())


main()
