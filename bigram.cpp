#include <iostream>
#include <vector>
#include <string>
#include <memory> //для юник птр
#include <fstream> //для чтения файлов и открытия
#include <sstream> //помогает работать со строчками (разбивает текст на слова)
#include <random> //генерит числа
#include <unordered_map> //позволяет искать данные по ключу
#include <optional> //для коробки

class DataLoader {
public:
    std::optional<std::string> loadText(const std::string& filepath) { //функция, которая принимает путь к файлу "filepath" и ретернит box, который может быть пустой, а может содержать строчку
        std::ifstream file(filepath); //создаю объект file, который открывет наш файл "filepath" (RAII)
        if (!file.is_open()) { //если файл не открывается по какой-то ошибке (опечатка, не может найти)
            std::cerr << "Error: Cannot open file '" << filepath << "'\n"; //cerr возваращет ошибку в компилятор красным цветом
            return std::nullopt; //std::optional, возвращает "пустую коробку" благодаря этому программа не падает, а сообщает, что данных нет
        }

        std::ostringstream buffer;//работаем с буффером для скорости
        buffer << file.rdbuf();//все данные из нашего файла переносим в буффер (мнгновенно)
        std::string content = buffer.str();//переделываем буффер в стринг

        if (content.empty()) {
            return std::nullopt; //также std::optional, если файл открылся, но в нем 0 букв, то он возвращает пустую коробку
        }

        return content; //возвращаем коробку, в которой лежит текст!!!!!!победа
    }
};

class LanguageModel { //класс является чертежем всех следующих моделей, так как планировалось mlp
public:
    virtual ~LanguageModel() {} //деструктор гарантирует, что, когда будем удалять модель, то не будет потери памяти, витруал правильно ее чистит

    virtual void train(const std::string& text) = 0;//в самом классе ничего не делает, но любой класс-наследник обязан написать свой код для этой функции, в нашем кейсе это обучение модели
    virtual std::string predictNext(const std::string& current_word) const = 0;

    std::string generateSentence(const std::string& start_word, int length) const { //общая функция для генерации текста
        std::string result = start_word; //начинаем предложение с первого слова ("I")
        std::string current = start_word;//делаем начальное слово нашем настоящим словом

        for (int i = 0; i < length - 1; ++i) { //цикл, чтобы программа вовремя перестала угадывать следующее слово
            std::string next = predictNext(current); //просим модель угадать след. слово
            if (next.empty()) break; //если моделька вернула пустоту, то просим прекратить этот цикл
            result += " " + next; //добавляем к начальному слову пробел и угаданное слово
            current = next; //теперь к конкретному слову добавляем пробел и новое угаданное слово (угаданное слово - текущее)
        }
        return result;
    }
};

class BigramModel : public LanguageModel {
private:
    std::unordered_map<std::string, std::unordered_map<std::string, int>> word_counts; //двойной словарь, в котором хранится сколько раз встречается какое-то слово. хранится все "I", открывается внут.словарь {am:5,love:15}
    mutable std::mt19937 gen{std::random_device{}()};//Это двигатель рулетки (генератор случайностей). Слово mutable нужно, потому что генератор должен прокручиваться (изменяться) при каждом броске, даже если функция у нас помечена как const (не изменяющая класс).

public:
    void train(const std::string& text) override {//оверрайд тут защищает от ошибок, комплитяор даже при опечатке будет понимать, что это train из родителя
        std::istringstream stream(text);//превращаем текст в потом, с++ сам будет разбивать текст на слова по пробелам
        std::string prev_word, curr_word;

        if (stream >> prev_word) { //считываем первое слово
            while (stream >> curr_word) { //пока в тексте есть слова, считываем их по одному
                word_counts[prev_word][curr_word]++; //идем в словарь и прибавляем единицу к счетчику этой пары
                prev_word = curr_word; //сдвигаемся вперед на одно слово
            }
        }
        std::cout << "Model trained successfully!:)\n";
    }

    std::string predictNext(const std::string& current_word) const override {
        auto it = word_counts.find(current_word);//ищем слово в нашем словаре.auto просит компилятор самому подставить сложный тип данных для итератора.

        if (it == word_counts.end() || it->second.empty()) {//если мы прошли до конца словаря и не нашли нужное слово, то ретерним пустоту
            return "";
        }

        std::vector<int> frequencies;//создаем два списка: 1) слова кандидаты
        std::vector<std::string> words; //2) как часто эти слова встречаются

        for (const auto& pair : it->second) { //перебираем все слова, которые шли после нашего, записываем их в списки
            words.push_back(pair.first);
            frequencies.push_back(pair.second);
        }

        std::discrete_distribution<> dist(frequencies.begin(), frequencies.end()); //создаем рулетку на основе частот (если слово встретилось 10 раз, его сектор в рулетке будет больше, чем у слова, которое встретилось 1 раз).
        return words[dist(gen)];//крутим рулетку dist(gen), получаем победный номер и возвращаем слово под этим номером!
    }
};

#ifndef HIDE_MAIN
int main() {
    DataLoader loader;

    std::optional<std::string> dataset = loader.loadText("bigram.txt")//пытаемся получить текст в коробочку

    if (!dataset.has_value()) { //проверяем, есть ли текст, если нет, то возвращаем 1
        std::cerr << "Dataset is empty or could not be loaded.\n";
        return 1;
    }
    auto my_model = std::make_unique<BigramModel>();
    my_model->train(dataset.value()); //достаем текст из коробки через .value() и отдаем модели на обучение

    std::vector<std::unique_ptr<LanguageModel>> models; //создаем вектор, который хранит указатели на базовый класс (полиморфизм, храним наследника в списке для базового класса)
    models.push_back(std::move(my_model));//юзаем стд мув, так как юник птр нельзя копировать, чтобы передать права владения на модель внутрь вектора

    std::cout << "\nGenerating text:\n";
    std::cout << models[0]->generateSentence("I", 50) << "\n"; //просим модель сгенерировать текст

    return 0;
}
#endif