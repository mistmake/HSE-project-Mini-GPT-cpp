#include <torch/torch.h>
#include <fstream>
#include <string>
#include <map>
#include <cmath>
#include "../config.h"
#include "../head/head.h"

namespace F = torch::nn::functional;
std::map<char, int> stoi; // encoder from symbol to number


struct MultiHeadAttentionImpl : torch::nn::Module {
    torch::nn::ModuleList heads = nullptr;
    torch::nn::Linear projection = nullptr;
    torch::nn::Dropout drop = nullptr;
    MultiHeadAttentionImpl(int heads_amount, int head_size) {
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
    MultiHeadAttentionImpl() = default;
    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> rawres;
        for (auto& raw : *heads) {
            auto h = raw->as<Head>();
            rawres.push_back(h->forward(x));
        }
        return drop(projection(torch::cat(rawres, -1)));
    }
};
TORCH_MODULE(MultiHeadAttention);