#include "funcs.h"
#include <algorithm>
#include <array>
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

#include "../TokenizerConfig.h"
#include "../utils/utils.h"

namespace fs = std::filesystem;

// read the text file and call on_line for each its line
std::optional<std::string> ReadDataset(const fs::path& file) {
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
}// read the text file and call on_line for each its line

void ProcessLine(const std::string& line, const TokenizerConfig& cfg, const std::vector<std::unique_ptr<WordFilter>>& word_filters, FreqMap& words, FreqMap& punct) {
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

// convert freq map to vector sorted by freq (decreasing)
std::vector<TokenFreq> SortByFreq(const FreqMap& map, std::uint64_t min_freq) {
    std::vector<TokenFreq> out;
    out.reserve(map.size());

    for (auto &[token, freq] : map) {
        if (freq >= min_freq) out.push_back({token, freq});
    }

    std::sort(out.begin(), out.end(), [](const TokenFreq& a, const TokenFreq& b) {
      if (a.freq != b.freq) {
        return a.freq > b.freq;
      } else {
        return a.token < b.token;
      }
    });

    return out;
}

// build freq subtokens in format (#...) (we need it because are going to develop the model into smth like T9)
std::vector<TokenFreq> BuildPieces(const std::vector<TokenFreq>& words, const TokenizerConfig& cfg) {
    FreqMap pieces;
    pieces.reserve(words.size() * 5);

    for (const auto & w : words) {
        const std::size_t len = w.token.size();
        if (len < cfg.min_piece_len) continue;

        for (std::size_t n = cfg.min_piece_len; n <= cfg.max_piece_len && n <= len; ++n) {
            pieces[w.token.substr(0, n)] += w.freq;
            for (std::size_t i = 1; i + n <= len; ++i) {
                pieces["#" + w.token.substr(i, n)] += w.freq;
            }
        }
    }

    return SortByFreq(pieces, cfg.min_piece_freq);
}


// build final token list with optional max_vocab limit
std::vector<TokenFreq> BuildFinalTokens(const std::vector<TokenFreq>& punct, const std::vector<TokenFreq>& words, const std::vector<TokenFreq>& pieces, std::optional<std::size_t> max_vocab) {
    std::vector<TokenFreq> out;
    std::unordered_set<std::string> seen;
    if (max_vocab.has_value()) {
        out.reserve(*max_vocab);
    }

    if (max_vocab.has_value()) {
        seen.reserve((*max_vocab) * 2 + 32);
    } else {
        seen.reserve((words.size() + pieces.size()) * 2 + 32);
    }

    auto add = [&](const std::string& token, std::uint64_t freq) {
        if (!token.empty() && (!max_vocab.has_value() || out.size() < *max_vocab) && seen.insert(token).second) {
            out.push_back({token, freq});
        }
    };

    for (const auto* special : kSpecialTokens) add(special, 0);


    for (const auto& t : punct) add(t.token, t.freq); // t - {string, freq} na we map it around punct
    for (const auto& t : words) add(t.token, t.freq); // ------//------
    for (const auto& t : pieces) add(t.token, t.freq);// ------//------
    return out;
}


// write token list file (one token per line for better work with them in the future)
bool WriteTokenList(const fs::path& output, const std::vector<TokenFreq>& tokens) {
    std::error_code ec;
    fs::create_directories(output.parent_path(), ec);

    std::ofstream out(output);
    if (!out) return false;
    for (const auto& t : tokens) out << t.token << '\n';
    return static_cast<bool>(out);
}
