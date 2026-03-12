#include "nanoArkadiiGPT.h"

#include <iostream>
#include <fstream>
#include <string>
#include <map>

#include "../config.h"
#include "../transformerblock/transformerblock.h"

namespace F = torch::nn::functional;

// the whole GPT!!!!
ArkadiiGPTImpl::ArkadiiGPTImpl() {
    //embedding table is bare context of symbols
    embedding_table = register_module(
        "embedding_table",
        torch::nn::Embedding(config.vocab_size, config.embed_dim_num)
        );
    // main head mixes
    main_head = register_module(
        "main_head",
        torch::nn::Linear(config.embed_dim_num, config.vocab_size)
        );
    //position embedding is thing that analyzes position of tokens to create better context
    position_embedding = register_module(
        "position_embedding",
        torch::nn::Embedding(config.block_size, config.embed_dim_num)
        );
    // sequential list of transformers(to make bigger model)
    transformers = register_module(
        "transformers",
        torch::nn::Sequential(
            TransformerBlock(config.embed_dim_num, config.head_number),
            TransformerBlock(config.embed_dim_num, config.head_number),
            TransformerBlock(config.embed_dim_num, config.head_number),
            TransformerBlock(config.embed_dim_num, config.head_number),
            TransformerBlock(config.embed_dim_num, config.head_number),
            TransformerBlock(config.embed_dim_num, config.head_number),
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.embed_dim_num}))
            )
        );
}

//bare prediction of next token using
//embedding, position embedding, main head and transformer blocks
torch::Tensor ArkadiiGPTImpl::forward(torch::Tensor& idx) {
    auto idxTime = idx.size(1);
    auto tokens_embed = embedding_table(idx);
    auto position_embed = position_embedding(
        torch::arange(
            idxTime,
            torch::TensorOptions().dtype(torch::kLong).device(idx.device())
            )
            );
    auto X = tokens_embed + position_embed;
    X = transformers->forward(X);
    auto logits = main_head(X);
    return logits;
}

//computes  the next token and loss function, returns std::pair
std::pair<torch::Tensor, torch::Tensor> ArkadiiGPTImpl::forward(const torch::Tensor& idx, torch::Tensor& targets) {
    auto idxTime = idx.size(1);
    auto tokens_embed = embedding_table(idx);
    auto position_embed = position_embedding(
        torch::arange(
            idxTime,
            torch::TensorOptions().dtype(torch::kLong).device(idx.device())
            )
            );
    auto X = tokens_embed + position_embed;
    X = transformers->forward(X);
    auto logits = main_head(X);
    auto B = logits.size(0);
    auto T = logits.size(1);
    auto C = logits.size(2);
    logits = logits.view({B*T, C});
    targets = targets.view({B*T});
    auto loss = F::cross_entropy(logits, targets);
    return {logits, loss};
}

// generates max_new_tokens tokens and concantenates with previous context matrix
torch::Tensor ArkadiiGPTImpl::generate(torch::Tensor idx, int max_new_tokens) {
    for (int i = 0; i < max_new_tokens; ++i) {
        auto idx_sliced = idx.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(-config.block_size, torch::indexing::None)
        });
        auto logits = forward(idx_sliced);
        logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
        auto probs = F::softmax(logits, F::SoftmaxFuncOptions(-1));
        auto pred = torch::multinomial(probs, 1);
        //auto pred = torch::argmax(probs, 1).unsqueeze(1);
        idx = torch::cat({idx, pred}, 1);
    }
    return idx;
}

//gets batch of dataset and returns coded tensor
//returned tensor size is (batch_size, block_size)
std::pair<torch::Tensor, torch::Tensor> get_batch(torch::Tensor data) {

    auto x = torch::zeros({config.batch_size, config.block_size}, torch::TensorOptions(torch::kInt64));
    auto y = torch::zeros({config.batch_size, config.block_size}, torch::TensorOptions(torch::kInt64));
    for (int i = 0; i < config.batch_size; ++i) {
        int start = torch::randint(0, data.numel() - config.block_size - 1, 1).item<int>();
        for (int j = 0; j < config.block_size; ++j) {
            x[i][j] = data[start + j];
            y[i][j] = data[start + j + 1];
        }
    }
    return {x, y};
}

//couts int amount tokens
void generate_and_cout(ArkadiiGPT& model, int amount) {
    auto raw = model->generate(torch::zeros(
        {1, 1}, torch::TensorOptions(torch::kLong).device(config.device)),
        amount
        );
    raw =raw.to(torch::kCPU);
    for (int i = 0; i < raw.numel(); ++i) {
        std::cout << config.vocab[raw[0][i].item<int>()];
    }
    std::cout << '\n';
}
