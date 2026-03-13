#include "utils.h"

bool MaxWordLenFilter::keep(const std::string& word) const  {
    return word.size() <= max_len;
}

ScopedTimer::~ScopedTimer(){
    const auto end = std::chrono::steady_clock::now();
    out_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
}

bool IsWord(unsigned const char c) {
    return std::isalnum(c) || c == '\'' || c == '-' || c == '_'; // alnum is alphanumeric char
}

bool IsPunct(unsigned char c) {
    return std::ispunct(c) != 0; // punct is punctuation char
}