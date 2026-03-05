#include <iostream>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <stdexcept>
#include <fstream>
#include <sstream>

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
            if (next == ' ') break; // Останавливаемся, если модель возвращает пробел (не знает что делать)
            result += next; // Приклеиваем к результату
            current = next; // Новая буква становится текущей для следующего шага
        }
        return result;
    }
};
class BigramModel : public LanguageModel { //класс - наследник
private:
    //сетка, в которой храню данные
    std::vector<std::vector<int>> counts; //кол-во, которое даст нам понять, сколько у нас повторяется тех или иных пар (пример на "hello")

public:
    BigramModel() {
        counts.resize(VOCAB_SIZE, std::vector<int>(VOCAB_SIZE, 0)); //конструктор, который делает сетку размером 256*256, зап     олненную 0
    }
    void train(const std::string& text) override { //составляю пары
        if (text.length() < 2) return; //ставлю условие, что мне нужно мин две буквы для пары
        for (size_t i = 0; i < text.length() - 1; ++i) { //считаю, сколько пар у меня во всем датасете
            unsigned char current_char = text[i];
            unsigned char next_char = text[i + 1];

            counts[current_char][next_char]++; //повышаю кол-во на 1
        }
        std::cout << "Model trained successfully on " << text.length() << " characters!\n";
    }
    char predictNext(char current) const override { //функция, которая предугатывает некст букву
        unsigned char curr = current;
        int max_count = -1;
        char best_next_char = ' ';
        for (size_t i = 0; i < VOCAB_SIZE; ++i) { //просматривает все 256 возможных след. букв, чтобы найти кол-во комбинаций, кот. больше повторяются по сравнению с другими
            if (counts[curr][i] > max_count) {
                max_count = counts[curr][i];
                best_next_char = static_cast<char>(i);
            }
        }
        if (max_count == 0) { //если моделька никогда эту букву не видела, то она возвращает пробел
            return ' ';
        }
        return best_next_char;
    }
};

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
            std::cout << "\nGenerating text starting with 't':\n";
            std::string output = pipeline[0]->generateSentence('t', 20);
            std::cout << output << "\n";

        } else {
            std::cout << "The dataset file is empty.\n";
        }
    } catch (const FileIOException& error) {
        std::cerr << "CRITICAL ERROR: " << error.what() << "\n";
        std::cerr << "Please make sure you created the 'bigrame.txt' file!\n";
    }

    return 0;
}