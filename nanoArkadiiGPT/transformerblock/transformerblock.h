#pragma once
#include <torch/torch.h>
#include <fstream>
#include <string>
#include <cmath>
#include "../config.h"
#include "../multihead_attention/multihead_attention.h"

//feed forward is made in order to remove linearity between layers
//this is used to approximate every func in neural network
struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Sequential mod = nullptr;
    FeedForwardImpl() = default;
    FeedForwardImpl(int embed_dim_num);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(FeedForward);

// transformer block is used to combine all
struct TransformerBlockImpl : torch::nn::Module {
    MultiHeadAttention att = nullptr;
    FeedForward feed = nullptr;
    torch::nn::LayerNorm ln1 = nullptr;
    torch::nn::LayerNorm ln2 = nullptr;

    TransformerBlockImpl() = default;

    TransformerBlockImpl(int embed_dim_num, int head_amount);
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(TransformerBlock);