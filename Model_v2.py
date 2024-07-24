import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import math
import math, copy

np.random.seed(0)
torch.manual_seed(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=128, max_len=316):
        super(PositionalEncoding, self).__init__()
        # initialize position encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x for input after embbeding
        """
        x = x + self.pe[:, : x.size(1)]
        return x

class SequencePooling(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Y_hat = self.softmax(self.linear(x))
        x = x * Y_hat
        return x

# class InputConv(nn.Module):
#     def __init__(self):
#         super().__init__()
#         
#         # input_shape  = [1, 23, 256]
#         # output_shape = [64, 21, 254]
#         self.cnn1 = nn.Conv2d(
#             in_channels=1,
#             out_channels=64,
#             kernel_size=(3, 3),
#             stride=(1, 1),
#             padding="valid",
#         )
#         self.relu1 = nn.ReLU()  # activation
#         # output_shape = [64, 11, 127]
#         self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
#         # output_shape = [128, 9, 125]
#         self.cnn2 = nn.Conv2d(
#             in_channels=64,
#             out_channels=128,
#             kernel_size=(3, 3),
#             stride=(1, 1),
#             padding="valid",
#         )
#         self.relu2 = nn.ReLU()  # activation
#         # output_shape = [128, 5, 63]
#         self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
# 
#     def forward(self, x):
#         B,A,V,C = x.shape
#         x = self.cnn1(x)
#         x = self.relu1(x)
#         x = self.maxpool1(x)
#         x = self.cnn2(x)
#         x = self.relu2(x)
#         x = self.maxpool2(x)
#         x = torch.transpose(x, 3, 1)
#         # output_shape = [315, 128]
#         x = x.reshape(B, 315, -1)
# 
#         return x
class InputConv(nn.Module):
    def __init__(self):
        super().__init__()

        # input_shape  = [1, 23, 256]
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=64,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="valid",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)),
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding="valid",
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        )
        # output_shape = [64, 21, 254]

    def forward(self, x):
        B,A,V,C = x.shape
        x = self.net(x)
        x = torch.transpose(x, 3, 1)
        # output_shape = [315, 128]
        x = x.reshape(B, 315, -1)

        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads, mask=None) -> None:
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.heads = heads

        self.mask = mask

        torch._assert(d_model % heads == 0, f"d_model % heads == 0")

        self.d_k = d_model // heads
        self.d_q = d_model // heads
        self.d_v = d_model // heads

        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, x):
        # input size: [b, d_model, sequence_len]
        B, sequence_len, d_model = x.shape
        key     = self.W_k(x)
        query   = self.W_k(x)
        value   = self.W_k(x)

        # split heads
        key     = key  .reshape(B, sequence_len, self.heads, self.d_k).transpose(1, 2)
        query   = query.reshape(B, sequence_len, self.heads, self.d_q).transpose(1, 2)
        value   = value.reshape(B, sequence_len, self.heads, self.d_v).transpose(1, 2)

        attention_weight = torch.matmul(query, key.transpose(-1, -2)) / (self.d_k ** (1/2))

        if self.mask is not None:
            attention_weight += self.mask

        # size: [batch, head, sequence_len, d_k]
        scale_dot_product_attention = torch.matmul(
            F.softmax(
                torch.matmul(query, key.transpose(-1, -2)) / (self.d_k ** (1/2)),
                dim=3
            ), 
            value
        )

        # combine head
        attention = scale_dot_product_attention.transpose(1, 2).reshape(B, sequence_len, self.d_model)
    
        output = self.W_o(attention)
        return output

class FFN(nn.Module):
    def __init__(self, in_out_dim, inner_dim, dropout=0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim, inner_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, in_out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, d_model, inner_dim=2048, dropout=0.2) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, heads=16)
        self.ffn = FFN(in_out_dim=d_model, inner_dim=inner_dim, dropout=dropout)

    def forward(self, x):
        after_layernorm1 = self.layernorm1(x + self.multi_head_attention(x))
        output = self.layernorm2(after_layernorm1 + self.ffn(after_layernorm1))
        return output

class LCT(nn.Module):
    def __init__(self, d_model=128, inner_dim=2048, dropout=0.1):
        super(LCT, self).__init__()

        # 1) CNN preprocess
        self.input_conv = InputConv()

        # 2) Add the classification token
        # self.embed_dim = torch.rand(batch_size, 1, embed_dim)
        self.class_token = nn.Parameter(torch.rand(1, 1, d_model))

        # 3) positional encoding
        self.positionalencoding = PositionalEncoding(d_model=d_model)

        # 4) encoder
        self.encoders = nn.Sequential(
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout),
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout),
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout),
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout),
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout),
            Encoder(d_model=d_model, inner_dim=inner_dim, dropout=dropout)
        )

        # 5) sequence_pooling
        self.seq = SequencePooling(d_model=d_model)

        # 6) MLP
        # output shape
        # [inter-ictal, ictal]
        self.MLP = nn.Sequential(
            nn.Linear(in_features=128, out_features=2)
        )

    def forward(self, x):
        tokens = self.input_conv(x)
        B, N, PC = tokens.shape
        cls_tokens = self.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.positionalencoding(tokens)
        tokens = self.encoders(tokens)
        tokens = tokens[:, 0]
        output = self.MLP(tokens)

        return output