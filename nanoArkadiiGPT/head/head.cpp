#pragma once
#include <torch/torch.h>
#include <fstream>
#include <string>
#include <cmath>
#include "../config.h"

namespace F = torch::nn::functional;

struct HeadImpl : torch::nn::Module {
    torch::nn::Linear query = nullptr;
    torch::nn::Linear key = nullptr;
    torch::nn::Linear value = nullptr;
    torch::Tensor mask = torch::Tensor(nullptr);
    torch::nn::Dropout drop = nullptr;

    HeadImpl() = default;
    HeadImpl(int head_size) {
        mask = register_buffer("mask", torch::tril(torch::ones({config.block_size, config.block_size})));
        query = register_module(
            "query",
            torch::nn::Linear(torch::nn::LinearOptions(config.embed_dim_num, head_size).bias(false))
            );
        key = register_module(
            "key",
            torch::nn::Linear(torch::nn::LinearOptions(config.embed_dim_num, head_size).bias(false))
            );
        value = register_module(
    "value",
    torch::nn::Linear(torch::nn::LinearOptions(config.embed_dim_num, head_size).bias(false))
    );
        drop = register_module(
            "drop",
            torch::nn::Dropout(config.dropout)
            );
    }
    torch::Tensor forward(torch::Tensor& x) {
        auto T = x.size(1);
        auto K = key(x);
        auto Q = query(x);
        auto V = value(x);
        torch::Tensor w = Q.matmul(K.transpose(-2, -1));
        w = w / sqrt(K.size(-1));
        w = w.masked_fill(
            mask.index({torch::indexing::Slice(0, T), torch::indexing::Slice(0, T)}) == 0,
            float(-INFINITY)
            );
        w = F::softmax(w, F::SoftmaxFuncOptions(-1));
        w = drop(w);
        auto res = w.matmul(V);
        return res;
    }
};
TORCH_MODULE(Head);