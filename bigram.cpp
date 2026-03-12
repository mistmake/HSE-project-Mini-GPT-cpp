#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map>
#include <optional>

class DataLoader {
public:
    std::optional<std::string> loadText(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "Error: Cannot open file '" << filepath << "'\n";
            return std::nullopt;
        }

        std::ostringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        if (content.empty()) {
            return std::nullopt;
        }

        return content;
    }
};

class LanguageModel {
public:
    virtual ~LanguageModel() {}

    virtual void train(const std::string& text) = 0;
    virtual std::string predictNext(const std::string& current_word) const = 0;

    std::string generateSentence(const std::string& start_word, int length) const {
        std::string result = start_word;
        std::string current = start_word;

        for (int i = 0; i < length - 1; ++i) {
            std::string next = predictNext(current);
            if (next.empty()) break;

            result += " " + next;
            current = next;
        }
        return result;
    }
};

class BigramModel : public LanguageModel {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_counts;
    mutable std::mt19937 gen{std::random_device{}()};

public:
    void train(const std::string& text) override {
        std::istringstream stream(text);
        std::string prev_word, curr_word;

        if (stream >> prev_word) {
            while (stream >> curr_word) {
                word_counts[prev_word][curr_word]++;
                prev_word = curr_word;
            }
        }
        std::cout << "Model trained successfully!\n";
    }

    std::string predictNext(const std::string& current_word) const override {
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
};

#ifndef HIDE_MAIN
int main() {
    DataLoader loader;

    std::optional<std::string> dataset = loader.loadText("bigram.txt");

    if (!dataset.has_value()) {
        std::cerr << "Dataset is empty or could not be loaded. Exiting.\n";
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