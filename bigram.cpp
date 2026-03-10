#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <random> //библиотека для случайных чисел
#include <cmath>

constexpr size_t VOCAB_SIZE = 256;

class FileIOException : public std::runtime_error {
public:
    explicit FileIOException(const std::string& path)
        : std::runtime_error("Cannot read file: " + path) {}
};

class DataLoader {
public:
    std::optional<std::string> loadText(const std::string& filepath) {
        std::ifstream file(filepath); //автоматом закрывает файл

        if (!file.is_open()) {
            throw FileIOException(filepath); //отбрасывает ошибку
        }
        std::ostringstream buffer;
        buffer << file.rdbuf();
        std::string content = buffer.str();

        if (content.empty()) {
            return std::nullopt; //файл открыт, но он пустой
        }

        return content; //ретернит строчку в "бокс"
    }
};

class LanguageModel {
public:
    virtual ~LanguageModel() { //деструктор, чтобы дети правильно очищали свою память
        std::cout << "[Base] LanguageModel destroyed.\n";
    }
    virtual void train(const std::string& text) = 0; //pure virtual functions
    virtual char predictNext(char current) const = 0;

    std::string generateSentence(char start_char, int length) const {
        std::string result = "";
        result += start_char; // Добавляем первую букву

        char current = start_char;
        for (int i = 0; i < length - 1; ++i) {
            char next = predictNext(current); // Угадываем следующую

            // УБРАЛИ строчку с break, чтобы модель генерировала длинный текст с пробелами!

            result += next; // Приклеиваем к результату
            current = next; // Новая буква становится текущей для следующего шага
        }
        return result;
    }
};

class BigramModel : public LanguageModel { //класс - наследник
private:
    //сетка, в которой храню данные
    std::vector<std::vector<int>> counts;
    //ДОБАВЛЕНО: Генератор случайных чисел
    //mutable нужен, чтобы генератор мог менять свое внутреннее состояние внутри const-функции
    mutable std::mt19937 gen;

public:
    BigramModel() {
        counts.resize(VOCAB_SIZE, std::vector<int>(VOCAB_SIZE, 0)); //конструктор
        //настраиваем стартовое значение (seed) для генератора
        std::random_device rd;
        gen = std::mt19937(rd());
    }

    void train(const std::string& text) override { //составляю пары
        if (text.length() < 2) return;
        for (size_t i = 0; i < text.length() - 1; ++i) {
            unsigned char current_char = text[i];
            unsigned char next_char = text[i + 1];

            counts[current_char][next_char]++; //повышаю кол-во на 1
        }
        std::cout << "Model trained successfully on " << text.length() << " characters!\n";
    }

    char predictNext(char current) const override { //функция, которая предугатывает некст букву
        unsigned char curr = current;

        // Берем строку с частотами для нашей буквы
        const auto& row = counts[curr];

        // Проверяем, есть ли вообще варианты продолжения (видели ли мы эту букву в тексте)
        bool has_options = false;
        for (int count : row) {
            if (count > 0) {
                has_options = true;
                break;
            }
        }

        if (!has_options) {
            return ' '; //если моделька никогда эту букву не видела, то она возвращает пробел
        }
        //создаем рулетку на основе частот (аналог torch.multinomial)
        std::discrete_distribution<> dist(row.begin(), row.end());
        //крутим рулетку и получаем индекс буквы
        int next_char_index = dist(gen);

        return static_cast<char>(next_char_index);
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
            //200 символов для генерации слов
            std::string output = pipeline[0]->generateSentence('t', 200);
            std::cout << output << "\n";

        } else {
            std::cout << "The dataset file is empty.\n";
        }
    } catch (const FileIOException& error) {
        std::cerr << "CRITICAL ERROR: " << error.what() << "\n";
        std::cerr << "The 'bigrame.txt' file doesn't created!\n";
    }

    return 0;
}
#endif