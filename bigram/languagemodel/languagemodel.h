#pragma once

#include <string>
#include <fstream>

class LanguageModel {
public:
    virtual ~LanguageModel() = default;

    virtual void train(const std::string& text) = 0;
    virtual std::string predictNext(const std::string& current_word) const = 0;

    std::string generateSentence(const std::string& start_word, int length) const;
};