import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> list[torch.Tensor]:
    # q, k and v are all size -> [batch_size, 8, sequence_length, 64]
    d_k = q.size()[-1]  # 64

    # k.transpose(-1, -2) -> [batch_size, 8, 64, sequence_length]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    # [batch_size, 8, sequence_length, sequence_length] -> precursor to self-attention matrix
    print(f"scaled.size() : {scaled.size()}")

    if mask is not None:
        # [sequence_length, sequence_length]
        print(f"------- ADDING MASK of shape {mask.size()} -------")
        scaled += mask  # Broadcasting add. So just the last N dimensions need to match

    # [batch_size, 8, sequence_length, sequence_length]
    attention = F.softmax(scaled, dim=-1)
    # [batch_size, 8, sequence_length, 64]
    values = torch.matmul(attention, v)

    return values, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = d_model // num_heads  # 512 // 8 = 64
        self.qkv_layer = nn.Linear(
            d_model, 3 * d_model
        )  # [512, 3 * 512] -> [512, 1536]
        self.linear_layer = nn.Linear(d_model, d_model)  # [512, 512]

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        print(f"x.size(): {x.size()}")  # [batch_size, sequence_length, 512]
        batch_size, sequence_length, _ = x.size()

        qkv = self.qkv_layer(x)
        # [batch_size, sequence_length, 1536]
        print(f"qkv.size(): {qkv.size()}")

        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )
        # [batch_size, sequence_length, 8, 3 * 64] -> [batch_size, sequence_length, 8, 192]
        print(f"qkv.size(): {qkv.size()}")

        qkv = qkv.permute(0, 2, 1, 3)
        # [batch_size, 8, sequence_length, 192]
        print(f"qkv.size(): {qkv.size()}")

        q, k, v = qkv.chunk(3, dim=-1)
        # Each are [batch_size, 8, sequence_length, 64]
        print(f"q.size(): {q.size()}, k.size(): {k.size()}, v.size(): {v.size()}")

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # values.size() -> [batch_size, 8, sequence_length, 64], attention.size() -> [batch_size, 8, sequence_length, sequence_length]
        print(f"values.size(): {values.size()}, attention.size():{ attention.size()}")

        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        # [batch_size, sequence_length, 8 * 64] -> [batch_size, sequence_length, 512]
        print(f"values.size(): {values.size()}")

        out = self.linear_layer(values)
        # [batch_size, sequence_length, 512] -> same size as input
        print(f"out.size(): {out.size()}")

        # Because `out` and `x` have the same size, we can cascade them one after
        # the other many times without disrupting code logic
        return out


class LayerNormalisation(nn.Module):
    def __init__(self, parameters_shape: list[int], eps: float = 1e-5):
        super(LayerNormalisation, self).__init__()
        self.parameters_shape = parameters_shape  # Along which dimension to perform layer norm (typically embedding dimension -> last dimension)
        self.eps = eps  # For numerical stability

        # gamma and beta are learnable parameters
        self.gamma = nn.Parameter(torch.ones(parameters_shape))  # [512]
        self.beta = nn.Parameter(torch.zeros(parameters_shape))  # [512]

    def forward(
        self, x: torch.Tensor  # [batch_size, sequence_length, 512]
    ) -> torch.Tensor:
        # Last dimension along which we want to perform layer norm
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]  # [-1]

        mean = x.mean(dim=dims, keepdims=True)
        # [batch_size, sequence_length, 1] because keepdims=True
        print(f"mean.size(): ({mean.size()})")

        var = ((x - mean) ** 2).mean(dim=dims, keepdims=True)
        # [batch_size, sequence_length, 1] because keepdims=True
        print(f"var.size(): ({var.size()})")

        std = (var + self.eps).sqrt()
        # [batch_size, sequence_length, 1]
        print(f"std.size(): ({std.size()})")

        y = (x - mean) / std
        # [batch_size, sequence_length, 512]
        print(f"y.size(): {y.size()}")

        # We want to make sure these normalised values are applicable across training set (not just this batch),
        # that's why we have learnable parameters gamma and beta
        out = self.gamma * y + self.beta
        # [batch_size, sequence_length, 512]
        print(f"out.size(): {out.size()}")

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.d_model = d_model  # 512
        self.hidden = hidden  # 2048
        self.drop_prob = drop_prob
        self.linear_layer1 = nn.Linear(d_model, hidden)  # [512, 2048]
        self.linear_layer2 = nn.Linear(hidden, d_model)  # [2048, 512]
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:  # [batch_size, sequence_length, 512]
        x = self.linear_layer1(x)
        # [batch_size, sequence_length, 2048]
        print(f"x.size() after 1st linear layer: {x.size()}")
        x = self.relu(x)
        # [batch_size, sequence_length, 2048]
        print(f"x.size() after activation: {x.size()}")
        x = self.dropout(x)
        # [batch_size, sequence_length, 2048]
        print(f"x.size() after dropout: {x.size()}")
        x = self.linear_layer2(x)
        # [batch_size, sequence_length, 512]
        print(f"x.size() after 2nd linear layer: {x.size()}")

        return x


class EncoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ffn_hidden: int, num_heads: int = 8, drop_prob: float = 0.1
    ):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.drop_prob = drop_prob

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual_x = x  # [batch_size, sequence_length, 512]
        print("------- ATTENTION 1 ------")
        x = self.attention(x, mask=None)  # [batch_size, sequence_length, 512]
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        x = self.norm1(x + residual_x)  # [batch_size, sequence_length, 512]
        print("------- DROPOUT 1 ------")
        x = self.dropout1(x)  # [batch_size, sequence_length, 512]

        residual_x = x  # [batch_size, sequence_length, 512]
        print("------- FEED FORWARD ------")
        x = self.ffn(x)  # [batch_size, sequence_length, 512]
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        x = self.norm2(x + residual_x)  # [batch_size, sequence_length, 512]
        print("------- DROPOUT 2 ------")
        x = self.dropout2(x)  # [batch_size, sequence_length, 512]

        # Shape remains unchanged from input `x` -> [batch_size, sequence_length, 512]
        # This output `x` is going to be so much more context aware than the input `x`
        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int = 8,
        drop_prob: float = 0.1,
        num_layers: int = 1,
    ) -> torch.Tensor:
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.ffn_hidden = ffn_hidden
        self.num_heads = num_heads
        self.drop_prob = drop_prob
        self.num_layers = num_layers

        self.layers = nn.Sequential(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # This output is going to be so much more context aware than the input after some training has commenced
        return self.layers(x)  # [batch_size, sequence_length, 512]
