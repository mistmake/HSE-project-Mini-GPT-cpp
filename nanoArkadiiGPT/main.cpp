#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include "ArkadiiGPT/nanoArkadiiGPT.h"
#include "config.h"



int main() {
    for (int i = 0; i < config.vocab.size(); ++i) {
        config.stoi[config.vocab[i]] = i; // making encoder
    }
    std::ifstream data("dataset.txt"); //dataset
    if (!data) {
        std::cout << "Error opening file\n";
        return 1;
    }
    std::vector<int64_t> encoded;
    char c;
    while (data.get(c)) {
        if (c == '\n') {
            continue;
        }
        encoded.push_back(config.stoi[c]); // encoding full dataset
    }

    //transforming from vector to torch::Tensor
    torch::Tensor fulldata = torch::tensor(encoded, torch::TensorOptions().dtype(torch::kLong));

    //creating model, transforming to GPU(if possible) and couting symbols before training
    ArkadiiGPT model;
    model->to(config.device);
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
            std::cout << loss.item() << ("  Model saved, number: " + std::to_string(config.version)) <<"\n";
            torch::save(model, std::string("versions/version") + std::to_string(config.version++) + std::string(".pt"));
        }
    }
    generate_and_cout(model, 100); //model test
}
