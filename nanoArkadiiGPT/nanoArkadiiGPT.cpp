#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
namespace F = torch::nn::functional;
constexpr int
batch_size = 4,
block_size = 8,
vocab_size = 65,
learning_amount = 100000;
std::string vocab = " !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
std::map<char, int> stoi;

struct BigramImpl : torch::nn::Module {
    torch::nn::Embedding embedding_table = nullptr;
    BigramImpl(int vocab_size) {
        embedding_table = register_module(
            "embedding_table",
            torch::nn::Embedding(vocab_size, vocab_size)
            );
    }
    torch::Tensor forward(torch::Tensor& idx) {
        auto logits = embedding_table(idx);
        return logits;
    }
    std::pair<torch::Tensor, torch::Tensor> forward(torch::Tensor& idx, torch::Tensor& targets) {
        auto logits = embedding_table(idx);
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
            auto logits = forward(idx);
            logits = logits.index({torch::indexing::Slice(), -1, torch::indexing::Slice()});
            auto probs = F::softmax(logits, F::SoftmaxFuncOptions(-1));
            auto pred = torch::multinomial(probs, 1);
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
    Bigram model(vocab_size);
    generate_and_cout(model, 100);
    auto optim = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(1e-3));
    for (int i = 0; i < learning_amount; ++i) {
        auto [xb, yb] = get_batch(train_data);
        auto [logits, loss] = model->forward(xb, yb);

        optim.zero_grad();
        loss.backward();
        optim.step();
        if (i == 0 or i == learning_amount - 1) {
            std::cout << loss.item() << "\n";
        }
    }
    generate_and_cout(model, 300);
}