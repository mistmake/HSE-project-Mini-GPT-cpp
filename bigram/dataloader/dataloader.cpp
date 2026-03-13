#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <optional>
#include "dataloader.h"

std::optional<std::string> DataLoader::loadText(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file '" << filepath << "'\n";
        return std::nullopt;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();

    if (content.empty()) {
        return std::nullopt;
    }

    return content;
}