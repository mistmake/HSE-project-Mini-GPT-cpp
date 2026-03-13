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
#include <variant>
#include <vector>

#include "config/TokenizerConfig.h"
#include "utils/utils.h"
#include "processing/funcs.h"

namespace fs = std::filesystem;

// token -> number of appearances in dataset <{string, freq}>
using FreqMap = std::unordered_map<std::string, std::uint64_t>;
// token -> int id (encode)
using TokenToId = std::unordered_map<std::string, std::size_t>;
// integer id -> token (decode)
using IdToToken = std::vector<std::string>;

// variant for parsed line items and std::visit
using ParsedItem = std::variant<std::string, char>;

int main() {
  // here i use shared_ptr processing and output block share one stats object
  // (it useful to use it here also because of lifetime duration at the same value as the whole program)
  // moreover, unique_ptr is not enough in this case as both owners need same object at the same moment
  auto stats = std::make_shared<Stats>();
  long long elapsed_ms = 0;

  try {
    if (tokcfg.dataset_path.empty()) {
      throw ConfigError("dataset_path is empty :( ");
    }
    if (!fs::is_regular_file(tokcfg.dataset_path)) {
      throw ConfigError("dataset file is not found :(  " + fs::absolute(tokcfg.dataset_path).string());
    }
    //second realization idea was through std::cerr



    FreqMap words;
    FreqMap punct;
    words.reserve(500000);
    punct.reserve(256);
    // unique_ptr and polymorphism here (as it was said to be used in the project)
    std::vector<std::unique_ptr<WordFilter> > word_filters;
    word_filters.push_back(std::make_unique<MaxWordLenFilter>(tokcfg.max_word_len));

    std::optional<std::size_t> max_vocab_opt = std::nullopt;
    if (tokcfg.max_vocab > 0) max_vocab_opt = tokcfg.max_vocab;


    {
      ScopedTimer timer(elapsed_ms);

      const auto dataset_opt = ReadDataset(tokcfg.dataset_path);
      if (!dataset_opt.has_value()) {
        throw DatasetReadError("failed to read dataset :( " + fs::absolute(tokcfg.dataset_path).string());
      }

      const std::string& dataset = *dataset_opt;
      std::size_t start = 0;
      while (start <= dataset.size()) {
        const std::size_t end = dataset.find('\n', start);
        if (end == std::string::npos) {
          ++stats->lines;
          ProcessLine(dataset.substr(start), tokcfg, word_filters, words, punct);
          break;
        }
        ++stats->lines;
        ProcessLine(dataset.substr(start, end - start), tokcfg, word_filters, words, punct);
        start = end + 1;
      }

      stats->dataset_read = true;

      const auto punct_sorted = SortByFreq(punct, 1);
      const auto words_sorted = SortByFreq(words, tokcfg.min_word_freq);
      const auto pieces_sorted = BuildPieces(words_sorted, tokcfg);
      const auto final_tokens = BuildFinalTokens(punct_sorted, words_sorted, pieces_sorted, max_vocab_opt);

      if (!WriteTokenList(tokcfg.token_list_output_path, final_tokens)) {
        throw WriteError("Failed to write token list: " + tokcfg.token_list_output_path.string());
      }

      stats->punct = punct_sorted.size();
      stats->words = words_sorted.size();
      stats->pieces = pieces_sorted.size();
      stats->final_tokens = final_tokens.size();
    }

  } catch (const ConfigError& e) {
    std::cerr << e.what() << '\n';
    return 666;
  } catch (const DatasetReadError& e) {
    std::cerr << e.what() << '\n';
    return 666;
  } catch (const WriteError& e) {
    std::cerr << e.what() << '\n';
    return 666;
  } catch (const std::exception& e) {
    std::cerr << "Unexpected error: " << e.what() << '\n';
    return 666;
  }

  // beautiful stats output (for checking of medium results)
  std::cout << "dataset = " << fs::absolute(tokcfg.dataset_path) << '\n'
            << "dataset_read = " << (stats->dataset_read ? 1 : 0) << '\n'
            << "token_list = " << fs::absolute(tokcfg.token_list_output_path) << "\n\n"
            << "lines = " << stats->lines << '\n'
            << "punct = " << stats->punct << '\n'
            << "words = " << stats->words << '\n'
            << "pieces = " << stats->pieces << '\n'
            << "final_tokens = " << stats->final_tokens << "\n\n"
            << "time_consumed_in_ms = " << std::fixed << std::setprecision(3) << elapsed_ms << '\n';
  return 0;
}
