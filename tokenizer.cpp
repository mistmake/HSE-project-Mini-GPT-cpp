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


// token -> number of appearances in dataset <{string, freq}>
using FreqMap = std::unordered_map<std::string, std::uint64_t>;

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
    std::size_t pieces = 0;

};

constexpr std::array kSpecialTokens = {"<pad>", "<unk>", "<bos>", "<eos>", "<tab>", "<nl>"};

//controll of not adding invisible simbols into our list of tokens
constexpr bool IsIgnoredControl(unsigned char c) {
  return c < 32 && c != '\t';
}

// abstract filter and polymorphism
struct WordFilter {
  virtual ~WordFilter() = default;
  virtual bool keep(const std::string& word) const = 0;
};

struct MaxWordLenFilter : WordFilter {
  explicit MaxWordLenFilter(std::size_t max_len) : max_len(max_len) {}

  bool keep(const std::string& word) const override {
    return word.size() <= max_len;
  }

  std::size_t max_len;
};


// read the text file and call on_line for each its line
std::optional<std::string> ReadDataset(const fs::path& file) {
  std::ifstream in(file, std::ios::binary); // binary - for working with any type of symbols in the same way (done for windows as we tought the system on it as well)
  if (!in) return std::nullopt;

  std::string text((std::istreambuf_iterator(in)), std::istreambuf_iterator<char>());
  if (text.empty()) return std::nullopt;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find('\n', start);
    if (end == std::string::npos) break; // npos - smth like symbol not found
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

// RAII timer
struct ScopedTimer {
  long long& out_ms;
  std::chrono::steady_clock::time_point start;

  explicit ScopedTimer(long long& out_ms) : out_ms(out_ms), start(std::chrono::steady_clock::now()) {}

  ~ScopedTimer() {
    const auto end = std::chrono::steady_clock::now();
    out_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  }
};


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
std::vector<TokenFreq> BuildPieces(const std::vector<TokenFreq>& words, const Config& cfg) {
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
  for (const auto& t : tokens) {
    out << t.token << '\n';
    return static_cast<bool>(out);
  }
}






int main() {
  const Config cfg{};
  // here i use shared_ptr processing and output block share one stats object
  // (it useful to use it here also because of lifetime duration at the same value as the whole program)
  // moreover, unique_ptr is not enough in this case as both owners need same object at the same moment
  auto stats = std::make_shared<Stats>();
  long long elapsed_ms = 0;

  try {
    if (cfg.dataset_path.empty()) {
      throw ConfigError("dataset_path is empty :( ");
    }
    if (!fs::is_regular_file(cfg.dataset_path)) {
      throw ConfigError("dataset file is not found :(  " + fs::absolute(cfg.dataset_path).string());
    }


    FreqMap words;
    FreqMap punct;
    words.reserve(500000);
    punct.reserve(256);
    // unique_ptr and polymorphism here
    std::vector<std::unique_ptr<WordFilter> > word_filters;
    // word_filters.push_back(std::make_unique<MaxWordLenFilter>(cfg.max_word_len));

    std::optional<std::size_t> max_vocab_opt = std::nullopt;
    if (cfg.max_vocab > 0) max_vocab_opt = cfg.max_vocab;


    {
      ScopedTimer timer(elapsed_ms);

      const auto dataset_opt = ReadDataset(cfg.dataset_path);
      if (!dataset_opt.has_value()) {
        throw DatasetReadError("failed to read dataset :( " + fs::absolute(cfg.dataset_path).string());
      }

      const std::string& dataset = *dataset_opt;
      std::size_t start = 0;
      while (start <= dataset.size()) {
        const std::size_t end = dataset.find('\n', start);
        if (end == std::string::npos) {
          ++stats->lines;
          ProcessLine(dataset.substr(start), cfg, word_filters, words, punct);
          break;
        }
        ++stats->lines;
        ProcessLine(dataset.substr(start, end - start), cfg, word_filters, words, punct);
        start = end + 1;
      }

      stats->dataset_read = true;

      const auto punct_sorted = SortByFreq(punct, 1);
      const auto words_sorted = SortByFreq(words, cfg.min_word_freq);
      const auto pieces_sorted = BuildPieces(words_sorted, cfg);
      const auto final_tokens = BuildFinalTokens(punct_sorted, words_sorted, pieces_sorted, max_vocab_opt);

      if (!WriteTokenList(cfg.token_list_output_path, final_tokens)) {
        throw WriteError("failed to write token list :( " + cfg.token_list_output_path.string());
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
    std::cerr << "unexpected error :( " << e.what() << '\n';
    return 666;
  }

  // beautiful stats output (for checking of medium results)
  std::cout << "dataset = " << fs::absolute(cfg.dataset_path) << '\n'
            << "dataset_read = " << (stats->dataset_read ? 1 : 0) << '\n'
            << "token_list = " << fs::absolute(cfg.token_list_output_path) << "\n\n"
            << "lines = " << stats->lines << '\n'
            << "punct = " << stats->punct << '\n'
            << "words = " << stats->words << '\n'
            << "pieces = " << stats->pieces << '\n'
            << "final_tokens = " << stats->final_tokens << "\n\n"
            << "time_consumed_in_ms = " << std::fixed << std::setprecision(3) << elapsed_ms << '\n';
  return 0;
}
