import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Conv1d, MaxPool1d, BatchNorm1d, ReLU, Sequential, AdaptiveAvgPool1d, Linear, \
    MultiheadAttention, Dropout, LayerNorm, ModuleList, TransformerEncoderLayer, TransformerEncoder

from nn_models.inception_time import InceptionModule
from nn_models.rel_pos_multi_head_attention import RelPosTransformerEncoderLayer

"""
https://keras.io/examples/timeseries/timeseries_classification_transformer/
https://www.geeksforgeeks.org/how-to-use-pytorchs-nnmultiheadattention/
"""


def get_params(modul):
    import numpy as np
    model_parameters = filter(lambda p: p.requires_grad, modul.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(params)


class PositionalEncoding(Module):
    def __init__(self, embed_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(1, max_len, embed_dim)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float) * (-math.log(10000.0) / embed_dim))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch_size, seq_len, embed_dim)
        x = x + self.pe[:, :x.size(1), :]
        return x


# class TransformerEncoderLayer(Module):
#     def __init__(self, embed_dim, num_heads, dim_feedforward=2048, dropout=0.1):
#         super().__init__()
#         self.self_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
#         # self.linear_1 = Linear(embed_dim, dim_feedforward)
#         self.dropout = Dropout(dropout)
#         # self.linear_2 = Linear(dim_feedforward, embed_dim)
#
#         # input shape: [n, c, l]
#         self.conv1d_1 = Conv1d(in_channels=embed_dim,
#                                out_channels=dim_feedforward,
#                                kernel_size=1,
#                                stride=1,
#                                bias=False)
#         self.conv1d_2 = Conv1d(in_channels=dim_feedforward,
#                                out_channels=embed_dim,
#                                kernel_size=1,
#                                stride=1,
#                                bias=False)
#
#         self.norm1 = LayerNorm(embed_dim)
#         self.norm2 = LayerNorm(embed_dim)
#         self.dropout1 = Dropout(dropout)
#         self.dropout2 = Dropout(dropout)
#
#     def forward(self, src):
#         """ input shape/output shape: [batch size, seq length, input_shape] """
#         x, _ = self.self_attn(src, src, src, need_weights=False)
#         x = src + self.dropout1(x)
#         x = self.norm1(x)
#         # src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
#         x = torch.transpose(x, 1, 2)
#         x_2 = self.conv1d_2(self.dropout(F.relu(self.conv1d_1(x))))
#         x = x + self.dropout2(x_2)
#         x = torch.transpose(x, 1, 2)
#         x = self.norm2(x)
#         return x


class BasicTransformer(Module):
    def __init__(self,
                 input_shape,
                 out_size,
                 seq_length,
                 embed_dim,
                 ff_dim,
                 num_transformer_blocks,
                 mlp_units,
                 dropout=0,
                 mlp_dropout=0,
                 **kwargs):
        super().__init__()
        self.example_input_array = torch.zeros(10, input_shape, seq_length)
        # self.embedding_conv = Conv1d(in_channels=input_shape,
        #                              out_channels=embed_dim,
        #                              kernel_size=32,  # TODO
        #                              stride=1,
        #                              padding="same",
        #                              bias=True)
        self.embedding_lin = Linear(input_shape, embed_dim)
        assert embed_dim % 4 == 0
        n_filters = embed_dim // 4
        # self.embedding_inc = InceptionModule(in_channels=input_shape,
        #                                      n_filters=n_filters,
        #                                      kernel_sizes=(17, 33, 65),
        #                                      bottleneck_channels=32)
        # self.pos_encoder = PositionalEncoding(embed_dim, seq_length)
        # encoder_layer = TransformerEncoderLayer(d_model=embed_dim,
        #                                         nhead=embed_dim // 16,
        #                                         dim_feedforward=ff_dim,
        #                                         dropout=dropout,
        #                                         batch_first=True)
        encoder_layer = RelPosTransformerEncoderLayer(d_model=embed_dim,
                                                      nhead=embed_dim // 16,
                                                      seq_length=seq_length,
                                                      dim_feedforward=ff_dim,
                                                      dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_transformer_blocks)

        self.global_avg_pool = AdaptiveAvgPool1d(1)
        self.mlp_part = ModuleList()
        input_dim = seq_length
        for dim in mlp_units:
            linear = Linear(input_dim, dim)
            dropout = Dropout(mlp_dropout)
            self.mlp_part.extend([linear, dropout])
            input_dim = dim

        self.output_linear = Linear(input_dim, out_size)

    def forward(self, x):
        # shape: [batch size, input_shape, seq length] -> [batch size, embed_dim, seq]
        # x = self.embedding_conv(x)  # if conv1d
        # x = self.embedding_inc(x)  # if inception block

        # shape: [batch size, embed_dim, seq length] -> [batch size, seq length, embed_dim]
        x = torch.transpose(x, 1, 2)
        x = self.embedding_lin(x)  # if linear

        # x = self.pos_encoder(x)
        x = self.transformer_encoder(x)

        # shape: [batch size, seq length, embed_dim] -> [batch size, seq length, 1]
        x = self.global_avg_pool(x)
        # shape: [batch size, seq length, 1] -> [batch size, seq length]
        x = torch.squeeze(x)

        for layer in self.mlp_part:
            x = layer(x)

        x = self.output_linear(x)
        return x


if __name__ == "__main__":
    _embed_dim = 2
    _num_heads = 2
    _head_size = 2
    _dropout = 0
    self_attn = MultiheadAttention(_embed_dim, num_heads=_embed_dim, kdim=_embed_dim, dropout=_dropout,
                                   batch_first=True)

    batch_size = 32
    sequence_length = 100
    _x = torch.rand(batch_size, sequence_length, _embed_dim)  # (sequence_length, batch_size, embed_dim)
    attn_output, attn_output_weights = self_attn(_x, _x, _x)

    print(attn_output.size(), attn_output_weights.size())

    tel = TransformerEncoderLayer(_embed_dim, _num_heads, _head_size, dim_feedforward=128, dropout=0)

    op = tel(_x)
    print(op.shape)
