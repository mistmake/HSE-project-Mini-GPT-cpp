#include <iostream>
#include <vector>
#include <string>
#include <memory>

constexpr size_t VOCAB_SIZE = 256;
class LanguageModel {
public:
    virtual ~LanguageModel() { //деструктор, чтобы дети правильно очищали свою память
        std::cout << "[Base] LanguageModel destroyed.\n";
    }
    virtual void train(const std::string& text) = 0; //pure virtual functions
    virtual char predictNext(char current) const = 0;
};
class BigramModel : public LanguageModel { //класс - наследник
private:
    //сетка, в которой храню данные
    std::vector<std::vector<int>> counts; //кол-во, которое даст нам понять, сколько у нас повторяется тех или иных пар (пример на "hello")

public:
    BigramModel() {
        counts.resize(VOCAB_SIZE, std::vector<int>(VOCAB_SIZE, 0)); //конструктор, который делает сетку размером 256*256, заполненную 0
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