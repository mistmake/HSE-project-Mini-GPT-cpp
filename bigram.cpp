#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <random>
#include <unordered_map> // ДОБАВЛЕНО: для создания словарей (хэш-таблиц) слов

class FileIOException : public std::runtime_error {
public:
    explicit FileIOException(const std::string& path)
        : std::runtime_error("Cannot read file: " + path) {}
};

class DataLoader {
public:
    static std::optional<std::string> loadText(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            throw FileIOException(filepath);
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
    virtual ~LanguageModel() {
        std::cout << "[Base] LanguageModel destroyed.\n";
    }

    // ИЗМЕНЕНО: Теперь принимаем и возвращаем std::string (СЛОВА)
    virtual void train(const std::string& text) = 0;
    [[nodiscard]] virtual std::string predictNext(const std::string& current_word) const = 0;

    // ИЗМЕНЕНО: Генерация работает со словами и вставляет пробелы
    std::string generateSentence(const std::string& start_word, int length) const {
        std::string result = start_word;
        std::string current = start_word;

        for (int i = 0; i < length - 1; ++i) {
            std::string next = predictNext(current);

            // Если модель не знает, что идет дальше (слово встретилось 1 раз в конце текста)
            if (next.empty()) break;

            result += " " + next; // Склеиваем слова пробелом!
            current = next;
        }
        return result;
    }
};

class BigramModel : public LanguageModel {
private:
    // ИЗМЕНЕНО: Сложный словарь.
    // Читается так: "Слово А" -> ("Слово Б" -> 5 раз, "Слово В" -> 2 раза)
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_counts;
    mutable std::mt19937 gen;

public:
    BigramModel() {
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    void train(const std::string& text) override {
        // ИЗМЕНЕНО: std::istringstream автоматически разбивает текст на слова по пробелам!
        std::istringstream stream(text);
        std::string prev_word, curr_word;

        // Читаем самое первое слово
        if (stream >> prev_word) {
            // Пока в тексте есть следующие слова, читаем их по очереди
            while (stream >> curr_word) {
                word_counts[prev_word][curr_word]++; // Повышаем счетчик пары слов
                prev_word = curr_word;               // Текущее слово становится предыдущим
            }
        }
        std::cout << "Model trained successfully on word pairs!\n";
    }

    std::string predictNext(const std::string& current_word) const override {
        // Ищем текущее слово в нашем словаре
        auto it = word_counts.find(current_word);

        // Если слова нет в словаре или после него никогда не было других слов
        if (it == word_counts.end() || it->second.empty()) {
            return "";
        }

        // Подготавливаем списки для рулетки
        std::vector<int> frequencies;
        std::vector<std::string> words;

        // it->second — это все слова, которые шли после current_word
        for (const auto& pair : it->second) {
            words.push_back(pair.first);    // Сохраняем само слово
            frequencies.push_back(pair.second); // Сохраняем, сколько раз оно встретилось
        }

        // Создаем рулетку на основе частот
        std::discrete_distribution<> dist(frequencies.begin(), frequencies.end());
        int next_word_index = dist(gen); // Крутим рулетку

        return words[next_word_index]; // Возвращаем выпавшее слово
    }
};

#ifndef HIDE_MAIN
int main() {
    DataLoader loader;
    std::string path = "bigram.txt";

    try {
        std::optional<std::string> dataset = loader.loadText(path);

        if (dataset.has_value()) {
            std::unique_ptr<LanguageModel> my_model = std::make_unique<BigramModel>();
            my_model->train(dataset.value());

            std::vector<std::unique_ptr<LanguageModel>> pipeline;
            pipeline.push_back(std::move(my_model));

            std::cout << "\nGenerating text starting with 'I':\n";
            // ИЗМЕНЕНО: Передаем стартовое слово (например "I" или "The") и просим 50 слов!
            std::string output = pipeline[0]->generateSentence("I", 50);
            std::cout << output << "\n";

        } else {
            std::cout << "The dataset file is empty.\n";
        }
    } catch (const FileIOException& error) {
        std::cerr << "CRITICAL ERROR: " << error.what() << "\n";
    }

    return 0;
}
#endif