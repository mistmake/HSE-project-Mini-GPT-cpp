#pragma once
#include <torch/torch.h>
#include <string>

//creates a self-attention head which stores Q, K, V linear layers to support context of text
struct HeadImpl : torch::nn::Module {
    torch::nn::Linear query = nullptr;
    torch::nn::Linear key = nullptr;
    torch::nn::Linear value = nullptr;
    torch::Tensor mask = torch::Tensor(nullptr);
    torch::nn::Dropout drop = nullptr;

    HeadImpl() = default;
    // registers layers in torch dynamic tree
    HeadImpl(int head_size);
    // computes attention using formula softmax((Q*K^T)/sqrt(d)) * V, based on "Attention is all you need"
    torch::Tensor forward(torch::Tensor& x);
};
TORCH_MODULE(Head);