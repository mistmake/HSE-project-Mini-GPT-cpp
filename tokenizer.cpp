#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>


namespace fs = std::filesystem;


// token -> number of appearances in dataset <{string, freq}>
using FreqMap = std::unordered_map<std::string, std::uint64_t>;
// token -> int id (encode)
using TokenToId = std::unordered_map<std::string, std::size_t>;
// integer id -> token (decode)
using IdToToken = std::vector<std::string>;

// token and frequency pair used in sorted outputs (struct just for convenience)
struct TokenFreq {
    std::string token;
    std::uint64_t freq;
};

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

// main project settings (the things we should change in order of change of dataset)
struct Config {
    // dataset input file path
    fs::path dataset_path = "/Users/karpukhin.simeon/Desktop/untitled2/tinystories_full.txt";

    // final token list output file
    fs::path token_list_output_path = "/Users/karpukhin.simeon/Desktop/untitled2/list_of_tokens.txt";

    bool lowercase = true;

    // 0 means that we will keep all useful tokens, otherwise keep first n tokens
    std::size_t max_vocab = 0;

    // keep tokens only if they appear at least this many times
    std::size_t min_word_freq = 3;
    std::size_t min_piece_freq = 9;

    // subword piece length
    std::size_t min_piece_len = 3;
    std::size_t max_piece_len = 15;

    // ignore very long words to because we do not need strange tokens
    std::size_t max_word_len = 30;
};