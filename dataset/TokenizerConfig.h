#pragma once
#include <array>
#include <chrono>
#include <filesystem>

namespace fs = std::filesystem;

// main project settings (the things we should change in order of change of dataset)
struct TokenizerConfig {
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

extern TokenizerConfig tokcfg; // configuration for tokenizer