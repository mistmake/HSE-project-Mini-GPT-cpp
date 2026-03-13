#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <optional>
#include "dataloader/dataloader.h"
#include "languagemodel/languagemodel.h"
#include "bigrammodel/bigrammodel.h"


// Защита от дублирования main при тестировании
#ifndef HIDE_MAIN
int main() {
    DataLoader loader;

    std::optional<std::string> dataset = loader.loadText("data/dataset.txt");

    if (!dataset.has_value()) {
        std::cerr << "Dataset is empty or could not be loaded.\n";
        return 1;
    }
    auto my_model = std::make_unique<BigramModel>();
    my_model->train(dataset.value());

    std::vector<std::unique_ptr<LanguageModel>> models;
    models.push_back(std::move(my_model));

    std::cout << "\nGenerating text:\n";
    std::cout << models[0]->generateSentence("I", 50) << "\n";

    return 0;
}
#endif