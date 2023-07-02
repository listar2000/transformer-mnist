import torch
from model import MultiHeadAttention


def test_multi_head_attention():
    dim_model = 512
    num_heads = 8
    batch_size = 64
    seq_len = 10

    # initialize a multi-head attention layer
    multi_head_attention_layer = MultiHeadAttention(dim_model=dim_model, num_heads=num_heads)

    # generate some random data
    Q = torch.randn((batch_size, seq_len, dim_model))
    K = torch.randn((batch_size, seq_len, dim_model))
    V = torch.randn((batch_size, seq_len, dim_model))

    # run the multi-head attention layer
    output, attention_scores = multi_head_attention_layer(Q, K, V)

    # check the output sizes
    assert output.size() == (batch_size, seq_len, dim_model)
    assert attention_scores.size() == (batch_size, num_heads, seq_len, seq_len)

    print("MultiHeadAttention test passed!")


if __name__ == "__main__":
    test_multi_head_attention()