#include <torch/torch.h>
#include <array>
#include <filesystem>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "ArkadiiGPT/nanoArkadiiGPT.h"
#include "config.h"

namespace fs = std::filesystem;

namespace {
fs::path find_data_dir(const fs::path& executable_path) {
    const fs::path current_dir = fs::current_path();
    const fs::path exe_dir = executable_path.empty() ? fs::path{} : executable_path.parent_path();

    const std::array<fs::path, 8> candidates = {
        current_dir / "data",
        current_dir.parent_path() / "data",
        current_dir.parent_path().parent_path() / "data",
        current_dir.parent_path().parent_path().parent_path() / "data",
        exe_dir / "data",
        exe_dir.parent_path() / "data",
        exe_dir.parent_path().parent_path() / "data",
        exe_dir.parent_path().parent_path().parent_path() / "data"
    };

    for (const auto& candidate : candidates) {
        std::error_code ec;
        if (fs::is_directory(candidate, ec) && !ec) {
            return candidate;
        }
    }

    return current_dir / "data";
}
}

int main(int argc, char* argv[]) {
    config.stoi.clear();
    for (int i = 0; i < static_cast<int>(config.vocab.size()); ++i) {
        config.stoi[config.vocab[i]] = i; // making encoder
    }

    const fs::path executable_path = argc > 0 ? fs::path(argv[0]) : fs::path{};
    const fs::path data_dir = find_data_dir(executable_path);
    const fs::path dataset_path = data_dir / "dataset.txt";
    const fs::path checkpoint_dir = data_dir / "versions";
    const fs::path final_model_path = data_dir / "model.pt";

    std::ifstream data(dataset_path); //dataset
    if (!data) {
        std::cerr << "Error opening dataset: " << fs::absolute(dataset_path) << '\n';
        return 1;
    }

    std::error_code ec;
    fs::create_directories(checkpoint_dir, ec);
    if (ec) {
        std::cerr << "Error creating checkpoint directory: " << fs::absolute(checkpoint_dir)
                  << " (" << ec.message() << ")\n";
        return 1;
    }

    std::vector<int64_t> encoded;
    std::size_t skipped_chars = 0;
    char c;
    while (data.get(c)) {
        if (c == '\n') {
            continue;
        }

        const auto it = config.stoi.find(c);
        if (it == config.stoi.end()) {
            ++skipped_chars;
            continue;
        }

        encoded.push_back(it->second); // encoding full dataset
    }

    if (encoded.size() <= static_cast<std::size_t>(config.block_size + 1)) {
        std::cerr << "Dataset is too small for block_size=" << config.block_size << '\n';
        return 1;
    }

    if (skipped_chars > 0) {
        std::cout << "Skipped " << skipped_chars << " unsupported characters while encoding.\n";
    }

    //transforming from vector to torch::Tensor
    torch::Tensor fulldata = torch::tensor(encoded, torch::TensorOptions().dtype(torch::kLong));

    //creating model, transforming to GPU(if possible) and couting symbols before training
    ArkadiiGPT model;
    model->to(config.device);
    model->train();
    generate_and_cout(model, 100);
    //creating AdamW optimizatior
    auto optim = torch::optim::AdamW(model->parameters(), torch::optim::AdamWOptions(3e-4));
    //training loop: predicting, computing loss function(cross-entropy), computing gradient
    // and update weights and bias
    for (int i = 0; i < config.learning_amount; ++i) {
        auto [xb, yb] = get_batch(fulldata);
        xb = xb.to(config.device);
        yb = yb.to(config.device);
        auto [logits, loss] = model->forward(xb, yb);

        optim.zero_grad();
        loss.backward();
        optim.step();
        if (i % 1000 == 0) { //every 1000 iteration saving the model
            const fs::path checkpoint_path = checkpoint_dir / ("version" + std::to_string(config.version++) + ".pt");
            std::cout << loss.item() << "  Model saved: " << checkpoint_path.string() << '\n';
            torch::save(model, checkpoint_path.string());
        }
    }
    torch::save(model, final_model_path.string());
    std::cout << "Final model saved to " << final_model_path.string() << '\n';

    model->eval();
    generate_and_cout(model, 100); //model test
    return 0;
}
