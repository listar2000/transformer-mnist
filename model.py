import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import reshape_mnist_images


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor):
    """
    Compute the scaled dot product attention for query Q, key K, and value V matrices
    :param Q: Query matrix of shape (batch_size, num_heads, seq_len, depth)
    :param K: Key matrix of shape (batch_size, num_heads, seq_len, depth)
    :param V: Value matrix of shape (batch_size, num_heads, seq_len, depth)
    :return:
    - output: Result of the attention computation, of shape (batch_size, num_heads, seq_len, depth)
    - att_scores: Attention scores, of shape (batch_size, num_heads, seq_len, seq_len)
    """
    dim_k = K.size(-1)
    assert dim_k == Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(dim_k)
    att_scores = F.softmax(scores, dim=-1)
    output = torch.matmul(att_scores, V)

    return output, att_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, dim_model: int, num_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        assert dim_model % num_heads == 0

        self.dim_model = dim_model
        self.dim_k = dim_model // num_heads
        self.num_heads = num_heads

        # define the matrices (MLPs) for Q, K, V
        self.W_q = nn.Linear(dim_model, dim_model)
        self.W_k = nn.Linear(dim_model, dim_model)
        self.W_v = nn.Linear(dim_model, dim_model)

        # define the matrix (MLP) for the final layer
        self.dropout = nn.Dropout(p=dropout)
        self.W_o = nn.Linear(dim_model, dim_model)

    def split_heads(self, x: torch.Tensor, batch_size: int):
        x = x.view(batch_size, -1, self.num_heads, self.dim_k)
        return x.transpose(1, 2)

    def forward(self, Q, K, V):
        batch_size = Q.size(0)

        Q = self.split_heads(self.W_q(Q), batch_size)  # (batch_size, num_heads, seq_len, depth)
        K = self.split_heads(self.W_k(K), batch_size)
        V = self.split_heads(self.W_v(V), batch_size)

        scores, attention_scores = scaled_dot_product_attention(Q, K, V)
        scores = self.dropout(scores)

        scores = scores.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_model)  # back to original shape

        output = self.W_o(scores)

        return output, attention_scores


class PointWiseFFN(nn.Module):
    def __init__(self, dim_model, dim_ff, dropout=0.1):
        super(PointWiseFFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(dim_model, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim_model)
        )

    def forward(self, x):
        return self.ffn(x)


class EncoderLayer(nn.Module):
    def __init__(self, dim_model: int, dim_ffn: int, num_heads: int, dropout: float = 0.1):
        super(EncoderLayer, self).__init__()
        self.multi_head_att = MultiHeadAttention(dim_model, num_heads, dropout)
        self.point_wise_ffn = PointWiseFFN(dim_model, dim_ffn, dropout)

        # layer norms and dropouts
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.layer_norm_1 = nn.LayerNorm(dim_model)
        self.layer_norm_2 = nn.LayerNorm(dim_model)

    def forward(self, x: torch.Tensor, mask=None):
        # perform multi-head attention
        x_att, _ = self.multi_head_att(x, x, x)  # self-attention, the input Q, K, V are all x
        x = self.layer_norm_1(x + self.dropout_1(x_att))

        # perform FFN
        x_ffn = self.point_wise_ffn(x)
        x = self.layer_norm_2(x + self.dropout_2(x_ffn))

        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers: int, dim_model: int = 512, dim_ffn: int = 2048, num_heads: int = 8,
                 dropout: float = 0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(dim_model, dim_ffn, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len: int = 784, dim_model: int = 512):
        super(PositionalEmbedding, self).__init__()
        init_embed = nn.init.xavier_normal_(torch.empty(max_len, dim_model))
        self.embed = nn.Parameter(init_embed, requires_grad=True)

    def forward(self, x: torch.Tensor):
        # Align the last dimension of `x` with the last dimension of `self.embed`
        x = x.expand(-1, -1, self.embed.size(-1))
        return x + self.embed


class ImageTransformer(nn.Module):
    def __init__(self, dim_model: int = 512, num_class: int = 10):
        super(ImageTransformer, self).__init__()
        self.positional_embedding = PositionalEmbedding(dim_model=dim_model)
        self.transformer_encoder = TransformerEncoder(num_layers=2, dim_model=dim_model)
        self.classification_head = nn.Linear(dim_model, num_class)

    def forward(self, images: torch.Tensor):
        # (batch_size, 784, 1)
        img_reshaped = reshape_mnist_images(images)
        img_embed = self.positional_embedding(img_reshaped)
        img_encoded = self.transformer_encoder(img_embed)
        # perform global average pooling
        img_pooled = img_encoded.mean(dim=1)
        output = self.classification_head(img_pooled)
        return output
