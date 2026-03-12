#include <torch/torch.h>
#include <fstream>
#include <string>
#include <cmath>
#include "../config.h"
#include "../multihead_attention/multihead_attention.h"


struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Sequential mod = nullptr;
    FeedForwardImpl() = default;
    FeedForwardImpl(int embed_dim_num) {
        mod = register_module(
            "mod",
            torch::nn::Sequential(
                torch::nn::Linear(embed_dim_num, embed_dim_num * 4),
                torch::nn::ReLU(), // deletes linearity
                torch::nn::Linear(embed_dim_num * 4, embed_dim_num),
                torch::nn::Dropout(config.dropout)
                )
            );
    }
    torch::Tensor forward(torch::Tensor x) {
        return mod->forward(x);
    }
};
TORCH_MODULE(FeedForward);

struct TransformerBlockImpl : torch::nn::Module {
    MultiHeadAttention att = nullptr;
    FeedForward feed = nullptr;
    torch::nn::LayerNorm ln1 = nullptr;
    torch::nn::LayerNorm ln2 = nullptr;

    TransformerBlockImpl() = default;

    TransformerBlockImpl(int embed_dim_num, int head_amount) {
        int head_size = embed_dim_num / head_amount;
        att = register_module(
            "att",
            MultiHeadAttention(head_amount, head_size)
            );
        feed = register_module(
            "feed",
            FeedForward(embed_dim_num)
            );
        ln1 = register_module(
            "ln1",
            torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            );
        ln2 = register_module(
        "ln2",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            );
    }
    torch::Tensor forward(torch::Tensor x) {
        // addition of MHA and FF to x in order to add residual connections
        x = x + att(ln1(x));
        x = x + feed(ln2(x));
        return x;
    }
};
TORCH_MODULE(TransformerBlock);