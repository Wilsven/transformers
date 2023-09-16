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
    print(f"scaled.size(): {scaled.size()}")

    if mask is not None:
        # [sequence_length, sequence_length]
        print(f"------- ADDING MASK of shape {mask.size()} ------")
        scaled += mask  # Broadcasting add. So just the last N dimensions need to match

    # [batch_size, 8, sequence_length, sequence_length]
    attention = F.softmax(scaled, dim=-1)
    # [batch_size, 8, sequence_length, 64]
    values = torch.matmul(attention, v)

    return values, attention


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
        self, x: torch.Tensor  # [batch_size, sequence_length, 512]
    ) -> torch.Tensor:
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
        # [batch_size, sequence_length, 512]
        print(f"out.size(): {out.size()}")

        # Because `out` and `x` have the same size, we can cascade them one after
        # the other many times without disrupting code logic
        return out


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8):
        super(MultiHeadCrossAttention, self).__init__()
        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.head_dim = d_model // num_heads  # 512 // 8 = 64
        self.kv_layer = nn.Linear(d_model, 2 * d_model)  # [512, 2 * 512] -> [512, 1024]
        self.q_layer = nn.Linear(d_model, d_model)  # [512, 512]
        self.linear_layer = nn.Linear(d_model, d_model)  # [512, 512]

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        print(f"x.size(): {x.size()}")  # [batch_size, sequence_length, 512]
        batch_size, sequence_length, _ = x.size()

        kv = self.kv_layer(x)
        # [batch_size, sequence_length, 2 * 512] -> [batch_size, sequence_length, 1024]
        print(f"kv.size(): {kv.size()}")

        q = self.q_layer(y)
        # [batch_size, sequence_length, 512]
        print(f"q.size(): {q.size()}")

        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        # [batch_size, sequence_length, 8, 2 * 64] -> [batch_size, sequence_length, 8, 128]
        print(f"kv.size(): {kv.size()}")

        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        # [batch_size, sequence_length, 8, 64]
        print(f"q.size(): {q.size()}")

        kv = kv.permute(0, 2, 1, 3)
        # [batch_size, 8, sequence_length, 128]
        print(f"kv.size(): {kv.size()}")

        q = q.permute(0, 2, 1, 3)
        # [batch_size, 8, sequence_length, 64]
        print(f"q.size(): {q.size()}")

        k, v = kv.chunk(2, dim=-1)
        # Each are [batch_size, 8, sequence_length, 64]
        print(f"k.size(): {k.size()}, v.size(): {v.size()}")

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        # values.size() -> [batch_size, 8, sequence_length, 64], attention.size() -> [batch_size, 8, sequence_length, sequence_length]
        print(f"values.size(): {values.size()}, attention.size():{ attention.size()}")

        values = values.reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        # [batch_size, sequence_length, 8 * 64] -> [batch_size, sequence_length, 512]
        print(f"values.size(): {values.size()}")

        out = self.linear_layer(values)
        # [batch_size, sequence_length, 512]
        print(f"out.size(): {out.size()}")

        return out


class DecoderLayer(nn.Module):
    def __init__(
        self, d_model: int, ffn_hidden: int, num_heads: int = 8, drop_prob: float = 0.1
    ):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model  # 512
        self.ffn_hidden = ffn_hidden  # 2048
        self.num_heads = num_heads  # 8
        self.drop_prob = drop_prob

        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.encoder_decoder_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalisation([d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, decoder_mask: torch.Tensor
    ) -> torch.Tensor:
        _y = y  # For skip/residual connections -> [batch_size, sequence_length, 512]
        print("------- MASKED ATTENTION ------")
        y = self.attention(y, mask=decoder_mask)  # [batch_size, sequence_length, 512]
        print("------- ADD AND LAYER NORMALIZATION 1 ------")
        y = self.norm1(y + _y)  # [batch_size, sequence_length, 512]
        print("------- DROPOUT 1 ------")
        y = self.dropout1(y)  # [batch_size, sequence_length, 512]

        _y = y  # [batch_size, sequence_length, 512]
        print("------- CROSS ATTENTION ------")
        y = self.encoder_decoder_attention(
            x, y, mask=None
        )  # [batch_size, sequence_length, 512]
        print("------- ADD AND LAYER NORMALIZATION 2 ------")
        y = self.norm2(y + _y)  # [batch_size, sequence_length, 512]
        print("------- DROPOUT 2 ------")
        y = self.dropout2(y)  # [batch_size, sequence_length, 512]

        _y = y  # [batch_size, sequence_length, 512]
        print("------- FEED FORWARD ------")
        y = self.ffn(y)  # [batch_size, sequence_length, 512]
        print("------- ADD AND LAYER NORMALIZATION 3 ------")
        y = self.norm3(y + _y)  # [batch_size, sequence_length, 512]
        print("------- DROPOUT 3 ------")
        y = self.dropout3(y)  # [batch_size, sequence_length, 512]

        # Shape remains unchanged from input `x` -> [batch_size, sequence_length, 512]
        # This output `x` is going to be so much more context aware than the input `x`
        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, mask = inputs
        for module in self._modules.values():
            y = module(x, y, mask)  # [batch_size, sequence_length, 512]
        return y


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float = 0.1,
        num_layers: int = 1,
    ):
        super(Decoder, self).__init__()
        self.d_model = d_model  # 512
        self.ffn_hidden = ffn_hidden  # 2048
        self.num_heads = num_heads  # 8
        self.drop_prob = drop_prob
        self.num_layers = num_layers

        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(self.num_layers)
            ]
        )

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        # `x`: [batch_size, sequence_length, 512]
        # `y`: [batch_size, sequence_length, 512]
        # `mask`: [sequence_length, sequence_length]
        return self.layers(x, y, mask)  # [batch_size, sequence_length, 512]
