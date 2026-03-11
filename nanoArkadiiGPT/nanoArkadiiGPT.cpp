#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <cmath>

namespace F = torch::nn::functional;

constexpr int
batch_size = 64,
block_size = 256,
vocab_size = 65,
learning_amount = 50000,
embed_dim_num = 384, // n_embd
head_number = 6;
constexpr float dropout = 0.2;
torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);

int version = 0;

std::string vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::map<char, int> stoi;

struct HeadImpl : torch::nn::Module {
    torch::nn::Linear query = nullptr;
    torch::nn::Linear key = nullptr;
    torch::nn::Linear value = nullptr;
    torch::Tensor mask = torch::Tensor(nullptr);
    torch::nn::Dropout drop = nullptr;

    HeadImpl() = default;
    HeadImpl(int head_size) {
        mask = register_buffer("mask", torch::tril(torch::ones({block_size, block_size})));
        query = register_module(
            "query",
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
            );
        key = register_module(
            "key",
            torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
            );
        value = register_module(
    "value",
    torch::nn::Linear(torch::nn::LinearOptions(embed_dim_num, head_size).bias(false))
    );
        drop = register_module(
            "drop",
            torch::nn::Dropout(dropout)
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
            torch::nn::Linear(embed_dim_num, embed_dim_num)
            );
        drop = register_module(
            "drop",
            torch::nn::Dropout(dropout)
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

struct FeedForwardImpl : torch::nn::Module {
    torch::nn::Sequential mod = nullptr;
    FeedForwardImpl() = default;
    FeedForwardImpl(int embed_dim_num) {
        mod = register_module(
            "mod",
            torch::nn::Sequential(
                torch::nn::Linear(embed_dim_num, embed_dim_num * 4),
                torch::nn::ReLU(),
                torch::nn::Linear(embed_dim_num * 4, embed_dim_num),
                torch::nn::Dropout(dropout)
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
        x = x + att(ln1(x));
        x = x + feed(ln2(x));
        return x;
    }
};
TORCH_MODULE(TransformerBlock);


struct BigramImpl : torch::nn::Module {

    torch::nn::Embedding embedding_table = nullptr;
    torch::nn::Linear main_head = nullptr;
    torch::nn::Embedding position_embedding = nullptr;
    torch::nn::Sequential transformers = nullptr;

    /// @param vocab_size -
    BigramImpl() {
        embedding_table = register_module(
            "embedding_table",
            torch::nn::Embedding(vocab_size, embed_dim_num)
            );
        main_head = register_module(
            "main_head",
            torch::nn::Linear(embed_dim_num, vocab_size)
            );
        position_embedding = register_module(
            "position_embedding",
            torch::nn::Embedding(block_size, embed_dim_num)
            );
        transformers = register_module(
            "transformers",
            torch::nn::Sequential(
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                TransformerBlock(embed_dim_num, head_number),
                torch::nn::LayerNorm(torch::nn::LayerNormOptions({embed_dim_num}))
            )
            );
    }
    torch::Tensor forward(torch::Tensor& idx) {
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

    ///
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& idx, torch::Tensor& targets) {
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

    torch::Tensor generate(torch::Tensor idx, int max_new_tokens) {
        for (int i = 0; i < max_new_tokens; ++i) {
            auto idx_sliced = idx.index({
                torch::indexing::Slice(),
                torch::indexing::Slice(-block_size, torch::indexing::None)
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
};

TORCH_MODULE(Bigram);

std::pair<torch::Tensor, torch::Tensor> get_batch( torch::Tensor data) {

    auto x = torch::zeros({batch_size, block_size}, torch::TensorOptions(torch::kInt64));
    auto y = torch::zeros({batch_size, block_size}, torch::TensorOptions(torch::kInt64));
    for (int i = 0; i < batch_size; ++i) {
        int start = torch::randint(0, data.numel() - block_size - 1, 1).item<int>();
        for (int j = 0; j < block_size; ++j) {
            x[i][j] = data[start + j];
            y[i][j] = data[start + j + 1];
        }
    }
    return {x, y};
}

void generate_and_cout(Bigram& model, int amount) {
    auto raw = model->generate(torch::zeros(
        {1, 1}, torch::TensorOptions(torch::kLong).device(device)),
        amount
        );
    raw =raw.to(torch::kCPU);
    for (int i = 0; i < raw.numel(); ++i) {
        std::cout << vocab[raw[0][i].item<int>()];
    }
    std::cout << '\n';
}
int main() {
    for (int i = 0; i < vocab.size(); ++i) {
        stoi[vocab[i]] = i;
    }
    std::ifstream data("../nanoArkadiiGPT/data/tinystories_sample.txt");
    if (!data) {
        std::cout << "Файл не открылся\n";
        return 1;
    }
    std::vector<int64_t> encoded;
    char c;
    while (data.get(c)) {
        if (c == '\n') {
            continue;
        }
        encoded.push_back(stoi[c]);
    }

    torch::Tensor fulldata = torch::tensor(encoded, torch::TensorOptions().dtype(torch::kLong));
    torch::Tensor test = torch::randint(
        0,
        vocab_size,
        {batch_size, block_size},
        torch::kInt64
        );

    Bigram model;
    model->to(device);
    generate_and_cout(model, 100);
    auto optim = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(3e-4));
    for (int i = 0; i < learning_amount; ++i) {
        auto [xb, yb] = get_batch(fulldata);
        xb = xb.to(device);
        yb = yb.to(device);
        auto [logits, loss] = model->forward(xb, yb);

        optim.zero_grad();
        loss.backward();
        optim.step();
        if (i % 1000 == 0) {
            std::cout << loss.item() << ("  Model saved, number: " + std::to_string(version)) <<"\n";
            torch::save(model, std::string("versions/version") + std::to_string(version++) + std::string(".pt"));
        }
    }
    generate_and_cout(model, 100);
}
