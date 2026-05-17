import torch

bs = 16
sl = 32
hidden_dim = 768
n_head = 8
head_size = int(hidden_dim / n_head)


half_head_size = head_size // 2
i = torch.arange(half_head_size)
theta = 10000 ** -(2 * i / head_size)

pos = torch.arange(sl)
pos_theta = (
    torch.outer(pos, theta).repeat_interleave(2, dim=-1).unsqueeze(0).unsqueeze(2)
)

x = torch.rand((bs, sl, n_head, head_size))
x_rotate = torch.stack([-x[:, :, :, 1::2], x[:, :, :, ::2]], dim=-1).reshape(
    bs, sl, n_head, -1
)

out = x * pos_theta.cos() + x_rotate * pos_theta.sin()

print(out.shape)
