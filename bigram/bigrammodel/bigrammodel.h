#pragma once

#include <string>
#include <random>
#include <unordered_map>
#include "../languagemodel/languagemodel.h"

class BigramModel : public LanguageModel {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_counts;
    mutable std::mt19937 gen{std::random_device{}()};
public:
    void train(const std::string& text) override;
    std::string predictNext(const std::string& current_word) const override;
};