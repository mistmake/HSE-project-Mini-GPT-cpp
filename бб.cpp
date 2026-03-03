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
