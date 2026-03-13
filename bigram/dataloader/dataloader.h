#pragma once

#include <string>
#include <optional>

class DataLoader {
public:
    std::optional<std::string> loadText(const std::string& filepath);
};