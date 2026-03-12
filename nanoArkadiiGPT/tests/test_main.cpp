#include <gtest/gtest.h>
#include <torch/torch.h>

#include "config.h"

Config config;

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    config.device = torch::kCPU;
    config.stoi.clear();
    for (int i = 0; i < static_cast<int>(config.vocab.size()); ++i) {
        config.stoi[config.vocab[i]] = i;
    }

    torch::manual_seed(0);
    return RUN_ALL_TESTS();
}
