#pragma once
#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>
namespace fs = std::filesystem;

#include "../config/TokenizerConfig.h"

// custom exceptions
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct DatasetReadError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct WriteError : std::runtime_error {
    using std::runtime_error::runtime_error;
};


// processing stats printed at the end
struct Stats {
    bool dataset_read = false;
    std::size_t lines = 0;
    std::size_t punct = 0;
    std::size_t words = 0;
    std::size_t final_tokens = 0;
    std::size_t pieces = 0;
};

constexpr std::array<const char*, 6> kSpecialTokens = {"<pad>", "<unk>", "<bos>", "<eos>", "<tab>", "<nl>"};

// abstract filter and polymorphism
struct WordFilter {
    virtual ~WordFilter() = default;
    virtual bool keep(const std::string& word) const = 0;
};

struct MaxWordLenFilter : WordFilter {
    explicit MaxWordLenFilter(std::size_t max_len) : max_len(max_len) {}
    std::size_t max_len;
    bool keep(const std::string& word) const override;
};
struct TokenFreq {
    std::string token;
    std::uint64_t freq;
};

// RAII timer
struct ScopedTimer {
    long long& out_ms;
    std::chrono::steady_clock::time_point start;

    explicit ScopedTimer(long long& out_ms) : out_ms(out_ms), start(std::chrono::steady_clock::now()) {}

    ~ScopedTimer();
};

constexpr bool IsIgnoredControl(unsigned char c) {
    return c < 32 && c != '\t';
}

bool IsWord(unsigned const char c);

bool IsPunct(unsigned char c);

