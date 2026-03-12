#pragma once
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <cmath>

namespace F = torch::nn::functional;

std::map<char, int> stoi; // encoder from symbol to number


// the whole GPT!!!!
struct ArkadiiGPTImpl : torch::nn::Module {

    torch::nn::Embedding embedding_table = nullptr;
    torch::nn::Linear main_head = nullptr;
    torch::nn::Embedding position_embedding = nullptr;
    torch::nn::Sequential transformers = nullptr;

    ArkadiiGPTImpl();
    //bare prediction of next token using
    //embedding, position embedding, main head and transformer blocks
    torch::Tensor forward(torch::Tensor& idx);

    //computes  the next token and loss function, returns std::pair
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& idx, torch::Tensor& targets);
    // generates max_new_tokens tokens and concantenates with previous context matrix
    torch::Tensor generate(torch::Tensor idx, int max_new_tokens);
};

TORCH_MODULE(ArkadiiGPT);

//gets batch of dataset and returns coded tensor
//returned tensor size is (batch_size, block_size)
std::pair<torch::Tensor, torch::Tensor> get_batch(torch::Tensor data);

//couts int amount tokens
void generate_and_cout(ArkadiiGPT& model, int amount);