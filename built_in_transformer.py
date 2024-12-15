import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from utils import *
# input size
# | d_model |
# +---------+--
# |         |
# |         | 
# |         | max
# |         | length
# |         |
# |         |
# |         |
# +---------+--

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500) -> None:
        super().__init__()
        self.PE = torch.zeros(max_len, d_model)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        
        div_term = torch.exp(
            10000 ** torch.arange(0, d_model, 2)
        )

        self.PE[:, 0::2] = torch.sin(position / div_term)
        self.PE[:, 1::2] = torch.cos(position / div_term)
        
        self.PE = nn.Parameter(self.PE.unsqueeze(0))
        self.PE.requires_grad = False
        
    def forward(self, x):
        x = x + self.PE[:, : x.shape[1]]
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
        scale_dot_product_attention = scale_dot_product_attention.transpose(1, 2).reshape(B, sequence_len, self.d_model)
    
        output = self.W_o(scale_dot_product_attention)
        return output

class FFN(nn.Module):
    def __init__(self, in_out_dim, inner_dim) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_out_dim, inner_dim),
            nn.ReLU(),
            nn.Linear(inner_dim, in_out_dim)
        )

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        self.multi_head_attention = MultiHeadAttention(d_model=d_model, heads=16)
        self.ffn = FFN(in_out_dim=d_model, inner_dim=2408)

    def forward(self, x):
        after_layernorm1 = self.layernorm1(self.multi_head_attention(x) + x)
        output = self.layernorm2(self.ffn(after_layernorm1) + after_layernorm1)
        return output

class SequencePooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Y_hat = self.softmax(self.linear(x))
        x = x * Y_hat
        return x
    
class InputConv(nn.Module):
    def __init__(self):
        # input_shape  = [1, 23, 256]
        super().__init__()
        # output_shape = [64, 21, 254]
        self.cnn1 = nn.Conv2d(
            in_channels=1,
            out_channels=64,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="valid",
        )
        self.relu1 = nn.ReLU()  # activation
        # output_shape = [64, 11, 127]
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # output_shape = [128, 9, 125]
        self.cnn2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding="valid",
        )
        self.relu2 = nn.ReLU()  # activation
        # output_shape = [128, 5, 63]
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

    def forward(self, x):
        B,A,V,C = x.shape
        x = self.cnn1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = torch.transpose(x, 3, 1)
        # output_shape = [315, 128]
        x = x.reshape(B, 315, -1)

        return x

class LCT(nn.Module):
    def __init__(self, d_model=128, dropout=0.1):
        super(LCT, self).__init__()

        # 1) CNN preprocess
        self.input_conv = InputConv()

        # 2) Add the classification token
        # self.embed_dim = torch.rand(batch_size, 1, embed_dim)
        self.class_token = nn.Parameter(torch.rand(1, 1, d_model))

        # 3) positional encoding
        self.positionalencoding = PositionalEncoding(d_model=d_model)

        # 4) encoder
        encoderlayer = nn.TransformerEncoderLayer(d_model=d_model, nhead=16)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoderlayer, num_layers=6)

        # 5) sequence_pooling
        self.seq = SequencePooling(d_model=d_model)

        # 6) MLP
        self.MLP = nn.Sequential(
            nn.Linear(in_features=128, out_features=2)
        )
        self.drop = nn.Dropout(p=dropout)
            # output
        # [inter-ictal, ictal]

    def forward(self, x):
        tokens = self.input_conv(x)
        B, N, PC = tokens.shape
        cls_tokens = self.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.positionalencoding(tokens)
        print(tokens.shape)
        print(tokens)
        a = tokens
        tokens = self.encoder(tokens)
        print("--------")
        print(tokens.shape)
        print(tokens)
        tokens = tokens[:, 0]
        tokens = self.seq(tokens)
        tokens = self.drop(tokens)
        output = self.MLP(tokens)

        return output, a
    
data = torch.rand((2, 1, 23, 256))
model = LCT()

dada = model(data)[1]



import torch
import torch.nn as nn

# 定義一些參數
d_model = 128  # 輸入特徵的維度
nhead = 16  # 自注意力機制的頭數
num_layers = 6  # Transformer 編碼器的層數
dim_feedforward = 2048  # 前饋神經網路的隱藏層維度
dropout = 0.1  # dropout 的比例

# 創建一個 Transformer 編碼器層
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='relu'  # 使用 ReLU 激活函數
)

# 創建一個 Transformer 編碼器
encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

# 假設輸入序列的長度為 10，批次大小為 1
seq_len = 10
batch_size = 1

# 創建一個輸入張量，形狀為 (seq_len, batch_size, d_model)
#input_tensor = torch.randn(seq_len, batch_size, d_model)
input_tensor = torch.randn(2, 316, 128)

# 將輸入張量傳遞給編碼器
output_tensor = encoder(dada.transpose(1, 2))

print("Input shape:", dada)
print("Output shape:", output_tensor)