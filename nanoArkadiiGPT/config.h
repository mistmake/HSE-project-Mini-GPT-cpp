#pragma once
#include <torch/torch.h>

struct Config {
    const int
        batch_size = 64, // batches are used to create multithreading, 1 batch computes 1 token at a time
        block_size = 256, // block_size shows the maximum amount of tokens that ArkadiiGPT can support in context
        vocab_size = 65, // amount of chars which are available for model to analyze and predict
        learning_amount = 50000, // number of training iterations(1000 iterations = 5-7 minutes on RTX 4070)
        embed_dim_num = 384, // number of dimensions of one single token
        head_number = 6; // number of Attention heads in MultiHeadAttention
    const float dropout = 0.2;
    // dropout is used to drop randomly some neurons, this is made in order to prevent overfitting
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU; // device where to train

    int version = 0;

    //all tokens which ArkadiiGPT can generate: it is saved to decode from number to symbol
    std::string vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
};

Config config;

