import torch
import torch.nn as nn
import numpy as np
import math
import math, copy

np.random.seed(0)
torch.manual_seed(0)


class LCT(nn.Module):
    def __init__(self, embed_dim=128):
        super(LCT, self).__init__()

        # 1) CNN preprocess
        self.CNN = CNN()

        # 2) Add the classification token
        # self.embed_dim = torch.rand(batch_size, 1, embed_dim)
        self.class_token = nn.Parameter(torch.rand(1, 1, embed_dim))

        # 3) positional encoding
        self.positionalencoding = positionalencoding()
        self.positionalencoding.requires_grad = False

        # 4) encoder
        self.encoder1 = encoder()
        self.encoder2 = encoder()
        self.encoder3 = encoder()
        self.encoder4 = encoder()
        self.encoder5 = encoder()
        self.encoder6 = encoder()

        # 5) sequence_pooling
        self.seq = sequence_pooling()

        # 5) MLP
        self.MLP = nn.Sequential(
            nn.Linear(in_features=128, out_features=2),
            #nn.Softmax(dim=-1)
        )

        # output
        # [inter-ictal, ictal]

    def forward(self, x):
        tokens = self.CNN(x)
        B, N, PC = tokens.shape
        cls_tokens = self.class_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.positionalencoding(tokens)
        tokens = self.encoder1(tokens)
        tokens = self.encoder2(tokens)
        tokens = self.encoder3(tokens)
        tokens = self.encoder4(tokens)
        tokens = self.encoder5(tokens)
        tokens = self.encoder6(tokens)
        tokens = tokens[:, 0]
        tokens = self.seq(tokens)
        output = self.MLP(tokens)

        return output


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=28, patch_size=4, in_channel=1, embed_dim=28):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        n_patches = (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = n_patches

        self.proj = nn.Conv2d(
            in_channels=in_channel,
            kernel_size=patch_size,
            stride=patch_size,
            out_channels=embed_dim,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # in [N, 1, 28, 28] => out [N, 28, 7, 7]
        x = self.proj(x)
        # in [N, 28, 7, 7]  => out [N, 28, 49]
        x = x.flatten(2)
        # in [N, 28, 49]    => out [N, 49, 28]
        x = x.transpose(1, 2)
        return x


class positionalencoding(nn.Module):
    def __init__(self, d_model=128, max_len=316):
        super(positionalencoding, self).__init__()
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
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x


class multiheadedattention(nn.Module):
    def __init__(self, heads=16, embed_dim=128):
        super(multiheadedattention, self).__init__()
        self.d_heads = embed_dim // heads
        self.d_model = embed_dim
        self.heads = heads
        self.scale = self.d_heads**0.5
        self.q = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.k = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.v = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.proj = nn.Linear(in_features=embed_dim, out_features=embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def scaled_dot_product_attention(self, Q, K, V):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_probs = self.softmax(attn_scores)
        output = torch.matmul(attn_probs, V)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.heads, self.d_heads).transpose(1, 2)

    def combine_heads(self, x):
        batch_size, _, seq_length, self.d_heads = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, x):

        Q = self.split_heads(self.q(x))
        K = self.split_heads(self.k(x))
        V = self.split_heads(self.v(x))

        attn_output = self.scaled_dot_product_attention(Q, K, V)

        output = self.proj(self.combine_heads(attn_output))
        return output


class encoder(nn.Module):
    def __init__(self, embed_dim=128, epsilon=1e-5):
        super(encoder, self).__init__()
        self.norm1 = eval("nn.LayerNorm")(embed_dim, eps=epsilon)
        self.MLP = MLP()
        self.multiHeadedAttention = multiheadedattention()
        self.norm2 = eval("nn.LayerNorm")(embed_dim, eps=epsilon)

    def forward(self, x):
        x = x + self.multiHeadedAttention(self.norm1(x))
        x = x + self.MLP(self.norm2(x))
        return x


class MLP(nn.Module):
    def __init__(self, in_features=128, hidden_features=64, out_features=128):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        #self.act = nn.Sigmoid()
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class sequence_pooling(nn.Module):
    def __init__(self, d_model=128):
        super().__init__()
        self.linear = nn.Linear(in_features=d_model, out_features=d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Y_hat = self.softmax(self.linear(x))
        x = x * Y_hat
        return x


class CNN(nn.Module):
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

