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

// processing stats printed at the end
struct Stats {
    bool dataset_read = false;
    std::size_t lines = 0;
    std::size_t punct = 0;
    std::size_t words = 0;
    std::size_t final_tokens = 0;
};

constexpr std::array<const char*, 6> kSpecialTokens = {"<pad>", "<unk>", "<bos>", "<eos>", "<tab>", "<nl>"};

constexpr bool IsIgnoredControl(unsigned char c) {
  return c < 32 && c != '\t';
}

// read the text file and call on_line for each its line
std::optional<std::string> ReadDatasetText(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) return std::nullopt;

  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  if (text.empty()) return std::nullopt;

  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find('\n', start);
    if (end == std::string::npos) break;
    start = end + 1;
  }
  return text;
}

bool IsWord(unsigned const char c) {
  return std::isalnum(c) || c == '\'' || c == '-' || c == '_'; // alnum is alphanumeric char
}

bool IsPunct(unsigned char c) {
  return std::ispunct(c) != 0; // punct is punctuation char
}

// variant for parsed line items and std::visit
using ParsedItem = std::variant<std::string, char>;

// parse a line into words and punctuation and update frequency maps
void ProcessLine(const std::string& line, const Config& cfg, const std::vector<std::unique_ptr<WordFilter>>& word_filters, FreqMap& words, FreqMap& punct) {
  std::string current;
  current.reserve(32);

  std::vector<ParsedItem> parsed;
  parsed.reserve(line.size() / 2 + 1);

  auto flush_word = [&]() {
    if (current.empty()) return;

    bool keep_word = true;
    for (const auto& f : word_filters) {
      if (!f->keep(current)) {
        keep_word = false;
        break;
      }
    }

    if (keep_word) parsed.push_back(current);
    current.clear();
  };

  for (unsigned char c : line) {
    if (IsIgnoredControl(c)) continue;

    if (IsWord(c)) {
      if (cfg.lowercase) {
        c = static_cast<unsigned char>(std::tolower(c));
      }
      current.push_back(static_cast<char>(c));
    } else {
      flush_word();
      if (IsPunct(c)) {
        parsed.push_back(static_cast<char>(c));
      }
    }
  }
  flush_word();

  for (const auto& item : parsed) {
    std::visit([&](const auto& value) {
          using ValueType = std::decay_t<decltype(value)>;
          if constexpr (std::is_same_v<ValueType, std::string>) {
            const std::string& word_token = value;
            words[word_token] += 1;
          } else {
            const char punctuation_char = value;
            const std::string punctuation_token(1, punctuation_char);
            punct[punctuation_token] += 1;
          }
        },
        item);
  }
}