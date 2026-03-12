#include "multihead_attention.h"

#include <torch/torch.h>
#include <fstream>
#include <string>
#include <map>
#include <cmath>

#include "../config.h"
#include "../head/head.h"

namespace F = torch::nn::functional;

// creates n heads which divides full embedding matrix into n pieces and computes attention
MultiHeadAttentionImpl::MultiHeadAttentionImpl(int heads_amount, int head_size) {
    heads = register_module(
        "heads",
        torch::nn::ModuleList()
        );
    for (int i = 0; i < heads_amount; ++i) {
        heads->push_back(Head(head_size));
    }
    projection = register_module(
        "projection",
        torch::nn::Linear(config.embed_dim_num, config.embed_dim_num)
        );
    drop = register_module(
        "drop",
        torch::nn::Dropout(config.dropout)
        );
}

// after computting attention, uses projection linear layer in order to mix the context and
// returns embed_dim_number size tensor
torch::Tensor MultiHeadAttentionImpl::forward(torch::Tensor x) {
    std::vector<torch::Tensor> rawres;
    for (auto& raw : *heads) {
        auto h = raw->as<Head>();
        rawres.push_back(h->forward(x));
    }
    return drop(projection(torch::cat(rawres, -1)));
}
