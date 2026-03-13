#include <string>
#include <fstream>
#include "languagemodel.h"

std::string LanguageModel::generateSentence(const std::string& start_word, int length) const {
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
