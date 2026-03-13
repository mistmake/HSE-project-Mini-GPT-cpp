#pragma once
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../config/TokenizerConfig.h"
#include "../utils/utils.h"

namespace fs = std::filesystem;
using FreqMap = std::unordered_map<std::string, std::uint64_t>;
using TokenToId = std::unordered_map<std::string, std::size_t>;
using IdToToken = std::vector<std::string>;

// variant for parsed line items and std::visit
using ParsedItem = std::variant<std::string, char>;

// read the text file and call on_line for each its line
std::optional<std::string> ReadDataset(const fs::path& file);
void ProcessLine(const std::string& line, const TokenizerConfig& cfg, const std::vector<std::unique_ptr<WordFilter>>& word_filters, FreqMap& words, FreqMap& punct);
std::vector<TokenFreq> SortByFreq(const FreqMap& map, std::uint64_t min_freq);
std::vector<TokenFreq> BuildPieces(const std::vector<TokenFreq>& words, const TokenizerConfig& cfg);
std::vector<TokenFreq> BuildFinalTokens(const std::vector<TokenFreq>& punct, const std::vector<TokenFreq>& words, const std::vector<TokenFreq>& pieces, std::optional<std::size_t> max_vocab);
bool WriteTokenList(const fs::path& output, const std::vector<TokenFreq>& tokens);

