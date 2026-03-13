#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>
#include "bigrammodel.h"

void BigramModel::train(const std::string& text) {
    word_counts.clear();
    std::istringstream stream(text);
    std::string prev_word, curr_word;

    if (stream >> prev_word) {
        while (stream >> curr_word) {
            word_counts[prev_word][curr_word]++;
            prev_word = curr_word;
        }
    }
    std::cout << "Model trained successfully!:)\n";
}

std::string BigramModel::predictNext(const std::string& current_word) const {
    auto it = word_counts.find(current_word);

    if (it == word_counts.end() || it->second.empty()) {
        return "";
    }

    std::vector<int> frequencies;
    std::vector<std::string> words;

    for (const auto& pair : it->second) {
        words.push_back(pair.first);
        frequencies.push_back(pair.second);
    }

    std::discrete_distribution<> dist(frequencies.begin(), frequencies.end());
    return words[dist(gen)];
}