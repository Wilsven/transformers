import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

START_TOKEN = "<START>"
END_TOKEN = "<END>"
PAD_TOKEN = "<PAD>"


def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def scaled_dot_product(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmal(attention, v)

    return values, attention


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_sequence_length: int):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i / self.d_model)
        position = torch.arange(self.max_sequence_length, dtype=float).reshape(-1, 1)
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)

        return PE


class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"

    def __init__(
        self,
        d_model: int,
        max_sequence_length: int,
        language_to_index: dict[str, int],
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PAD_TOKEN,
    ):
        super().__init__()
        self.vocab_size = len(language_to_index)
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.language_to_index = language_to_index
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token

    def batch_tokenize(self, batch: int, start_token: bool, end_token: bool):
        def tokenize(sentence, start_token, end_token):
            sentence_word_indicies = [
                self.language_to_index[token] for token in list(sentence)
            ]
            if start_token:
                sentence_word_indicies.insert(
                    0, self.language_to_index[self.start_token]
                )

            if end_token:
                sentence_word_indicies.append(self.language_to_index[self.end_token])

            for _ in range(len(sentence_word_indicies), self.max_sequence_length):
                sentence_word_indicies.append(
                    self.language_to_index[self.padding_token]
                )

            return sentence_word_indicies

        tokenized = []
        for sentence_num in range(len(batch)):
            tokenized.append(tokenize(batch[sentence_num]), start_token, end_token)

        tokenized = torch.stack(tokenized)

        return tokenized.to(get_device())

    def forward(self, x: torch.Tensor, start_token: bool, end_token: bool):  # sentence
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())
        x = self.dropout(x + pos)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model, 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size, sequence_length, _ = x.size()
        qkv = self.qkv_layer(x)
        qkv = qkv.reshape(
            batch_size, sequence_length, self.num_heads, 3 * self.head_dim
        )
        qkv = qkv.permute(0, 2, 1, 3)
        q, k, v = qkv.chunk(3, dim=-1)
        values, _ = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        out = self.linear_layer(values)

        return out


class LayerNormalisation(nn.Module):
    def __init__(self, parameters_shape: list[int], eps: float = 1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs: torch.Tensor):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdims=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdims=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, hidden: int, drop_prob: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x: torch.Tensor):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)
        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(
        self, x: torch.Tensor, self_attention_mask: Optional[torch.Tensor] = None
    ):
        residual_x = x.clone()
        x = self.attention(x, mask=self_attention_mask)
        x = self.norm1(x + residual_x)
        x = self.dropout1(x)

        residual_x = x.clone()
        x = self.ffn(x)
        x = self.norm2(x + residual_x)
        x = self.dropout2(x)

        return x


class SequentialEncoder(nn.Sequential):
    def forward(self, *inputs):
        x, self_attention_mask = inputs
        for module in self._modules.values():
            x = module(x, self_attention_mask)

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        num_layers: int,
        drop_prob: float,
        max_sequence_length: int,
        language_to_index: dict[str, int],
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PAD_TOKEN,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            d_model,
            max_sequence_length,
            language_to_index,
            start_token,
            end_token,
            padding_token,
        )
        self.layers = SequentialEncoder(
            *[
                EncoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        self_attention_mask: Optional[torch.Tensor] = None,
        start_token: bool = True,
        end_token: bool = True,
    ):
        x = self.sentence_embedding(x, start_token, end_token)
        x = self.layers(x, self_attention_mask)

        return x


class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.kv_layer = nn.Linear(d_model, 2 * d_model)
        self.q_layer = nn.Linear(d_model, d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(
        self, x: torch.Tensor, y: torch.Tensor, mask: Optional[torch.Tensor] = None
    ):
        # in practice, this is the same for both languages...so we can technically combine with normal attention
        batch_size, sequence_length, _ = x.size()
        kv = self.kv_layer(x)
        q = self.q_layer(y)
        kv = kv.reshape(batch_size, sequence_length, self.num_heads, 2 * self.head_dim)
        q = q.reshape(batch_size, sequence_length, self.num_heads, self.head_dim)
        kv = kv.permute(0, 2, 1, 3)
        q = q.permute(0, 2, 1, 3)
        k, v = kv.chunk(2, dim=-1)
        # We don't need the mask for cross attention, removing in outer function!
        values, _ = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3).reshape(
            batch_size, sequence_length, self.num_heads * self.head_dim
        )
        out = self.linear_layer(values)

        return out


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, ffn_hidden: int, num_heads: int, drop_prob: float):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNormalisation([d_model])
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.norm2 = LayerNormalisation([d_model])
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.norm3 = LayerNormalisation([d_model])
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
    ):
        residual_y = y.clone()
        y = self.self_attention(y, mask=self_attention_mask)
        y = self.norm1(y + residual_y)
        y = self.dropout1(y)

        residual_y = y.clone()
        y = self.cross_attention(x, y, mask=cross_attention_mask)
        y = self.norm2(y + residual_y)
        y = self.dropout2(y)

        residual_y = y.clone()
        y = self.ffn(y)
        y = self.norm3(y + residual_y)
        y = self.dropout3(y)

        return y


class SequentialDecoder(nn.Sequential):
    def forward(self, *inputs):
        x, y, self_attention_mask, cross_attention_mask = inputs
        for module in self._modules.values():
            y = module(x, y, self_attention_mask, cross_attention_mask)

        return y


class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        language_to_index: dict[str, int],
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PAD_TOKEN,
    ):
        super().__init__()
        self.sentence_embedding = SentenceEmbedding(
            d_model,
            max_sequence_length,
            language_to_index,
            start_token,
            end_token,
            padding_token,
        )
        self.layers = SequentialDecoder(
            *[
                DecoderLayer(d_model, ffn_hidden, num_heads, drop_prob)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        self_attention_mask: torch.Tensor,
        cross_attention_mask: torch.Tensor,
        start_token: bool,
        end_token: bool,
    ):
        y = self.sentence_embedding(y, start_token, end_token)
        y = self.layers(x, y, self_attention_mask, cross_attention_mask)

        return y


class Transformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        ffn_hidden: int,
        num_heads: int,
        drop_prob: float,
        num_layers: int,
        max_sequence_length: int,
        target_vocab_size: int,
        source_to_index: dict[str, int],
        target_to_index: dict[str, int],
        start_token: str = START_TOKEN,
        end_token: str = END_TOKEN,
        padding_token: str = PAD_TOKEN,
    ):
        super().__init__()
        self.encoder = Encoder(
            d_model,
            ffn_hidden,
            num_heads,
            num_layers,
            drop_prob,
            max_sequence_length,
            source_to_index,
            start_token,
            end_token,
            padding_token,
        )
        self.decoder = Decoder(
            d_model,
            ffn_hidden,
            num_heads,
            num_layers,
            drop_prob,
            max_sequence_length,
            target_to_index,
            start_token,
            end_token,
            padding_token,
        )
        self.linear = nn.Linear(d_model, target_vocab_size)
        self.device = get_device()

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        encoder_self_attention_mask: Optional[torch.Tensor] = None,
        decoder_self_attention_mask: Optional[torch.Tensor] = None,
        decoder_cross_attention_mask: Optional[torch.Tensor] = None,
        enc_start_token: bool = False,
        enc_end_token: bool = False,
        dec_start_token: bool = True,
        dec_end_token: bool = False,
    ):
        x = self.encoder(x, encoder_self_attention_mask, enc_start_token, enc_end_token)
        out = self.decoder(
            x,
            y,
            decoder_self_attention_mask,
            decoder_cross_attention_mask,
            dec_start_token,
            dec_end_token,
        )
        out = self.linear(out)

        return out
