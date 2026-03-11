#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <cmath>

namespace F = torch::nn::functional;

constexpr int
batch_size = 4,
block_size = 8,
vocab_size = 65,
learning_amount = 10000,
embed_dim_num = 32, // n_embd
head_size = 32;
auto device = (torch::cuda::is_available()) ? "cuda" : "cpu";

std::string vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

std::map<char, int> stoi;

struct HeadImpl : torch::nn::Module {
    torch::nn::Linear query = nullptr;
    torch::nn::Linear key = nullptr;
    torch::nn::Linear value = nullptr;
    torch::Tensor mask = torch::Tensor(nullptr);

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
    }

    torch::Tensor forward(torch::Tensor& x) {
        auto B = x.size(0);
        auto T = x.size(1);
        auto C = x.size(2);
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
        auto res = w.matmul(V);
        return res;

    }
};
TORCH_MODULE(Head);

struct MultiHeadAttentionImpl : torch::nn::Module {
    torch::nn::ModuleList heads = nullptr;
    MultiHeadAttentionImpl(int heads_amount, int head_size) {
        heads = register_module(
            "heads",
            torch::nn::ModuleList()
            );
        for (int i = 0; i < heads_amount; ++i) {
            heads->push_back(Head(head_size));
        }
    }
    MultiHeadAttentionImpl() = default;
    torch::Tensor forward(torch::Tensor x) {
        std::vector<torch::Tensor> rawres;
        for (auto& raw : *heads) {
            auto h = raw->as<Head>();
            rawres.push_back(h->forward(x));
        }
        return torch::cat(rawres, -1);
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
                torch::nn::Linear(embed_dim_num, embed_dim_num),
                torch::nn::ReLU()
                )
            );
    }
    torch::Tensor forward(torch::Tensor x) {
        return mod->forward(x);
    }
};
TORCH_MODULE(FeedForward);

struct BigramImpl : torch::nn::Module {

    torch::nn::Embedding embedding_table = nullptr;
    torch::nn::Linear main_head = nullptr;
    torch::nn::Embedding position_embedding = nullptr;
    MultiHeadAttention attention;

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
        attention = register_module(
            "attention_head",
            MultiHeadAttention(4, head_size/4)
            );

    }
    torch::Tensor forward(torch::Tensor& idx) {
        auto idxTime = idx.size(1);
        auto tokens_embed = embedding_table(idx);
        auto position_embed = position_embedding(torch::arange(idxTime));
        auto X = tokens_embed + position_embed;
        X = attention(X);
        auto logits = main_head(X);
        return logits;
    }

    ///
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& idx, torch::Tensor& targets) {
        auto idxTime = idx.size(1);
        auto tokens_embed = embedding_table(idx);
        auto position_embed = position_embedding(torch::arange(idxTime));
        auto X = tokens_embed + position_embed;
        X = attention(X);
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
    auto raw = model->generate(torch::zeros({1, 1}, torch::TensorOptions(torch::kLong)), amount);
    for (int i = 0; i < raw.numel(); ++i) {
        std::cout << vocab[raw[0][i].item<int>()];
    }
    std::cout << '\n';
}
int main() {
    for (int i = 0; i < vocab.size(); ++i) {
        stoi[vocab[i]] = i;
    }
    std::ifstream data("../nanoArkadiiGPT/data/Shakespeare.txt");
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

    torch::Tensor fulldata = torch::tensor(encoded);
    auto train_data = fulldata.slice(0, 0, static_cast<int>(fulldata.numel() * 0.9));
    auto test_data = fulldata.slice(0, static_cast<int>(fulldata.numel() * 0.9));
    torch::Tensor test = torch::randint(
        0,
        vocab_size,
        {batch_size, block_size},
        torch::kInt64
        );

    Bigram model;
    generate_and_cout(model, 100);
    auto optim = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(1e-3));
    for (int i = 0; i < learning_amount; ++i) {
        auto [xb, yb] = get_batch(fulldata);
        auto [logits, loss] = model->forward(xb, yb);

        optim.zero_grad();
        loss.backward();
        optim.step();
        if (i == 0 or i == learning_amount - 1) {
            std::cout << loss.item() << "\n";
        }
    }
    generate_and_cout(model, 100);
}
