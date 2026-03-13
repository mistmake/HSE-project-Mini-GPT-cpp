#pragma once

#include <torch/torch.h>
#include <fstream>
#include <string>
#include <cmath>

namespace F = torch::nn::functional;

struct MultiHeadAttentionImpl : torch::nn::Module {
    torch::nn::ModuleList heads = nullptr;
    torch::nn::Linear projection = nullptr;
    torch::nn::Dropout drop = nullptr;
    // creates n heads which divides full embedding matrix into n pieces and computes attention
    MultiHeadAttentionImpl(int heads_amount, int head_size);
    MultiHeadAttentionImpl() = default;
    // after computting attention, uses projection linear layer in order to mix the context and
    // returns embed_dim_number size tensor
    torch::Tensor forward(torch::Tensor x);
};
TORCH_MODULE(MultiHeadAttention);
