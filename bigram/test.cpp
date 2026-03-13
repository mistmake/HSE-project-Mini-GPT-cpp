#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

#include "bigrammodel/bigrammodel.h"
#include "dataloader/dataloader.h"

TEST(DataLoaderTest, ReturnsNulloptWhenFileNotFound) {
    DataLoader loader;
    auto result = loader.loadText("fake_missing_file.txt");
    EXPECT_FALSE(result.has_value());
}

TEST(DataLoaderTest, ReadsExistingFile) {
    const auto path = std::filesystem::temp_directory_path() / "bigram_loader_test.txt";
    {
        std::ofstream out(path);
        out << "hello world";
    }

    DataLoader loader;
    auto result = loader.loadText(path);

    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "hello world");

    std::filesystem::remove(path);
}

TEST(BigramModelTest, PredictsCorrectWord) {
    BigramModel model;
    model.train("hello world hello world hello world");
    EXPECT_EQ(model.predictNext("hello"), "world");
}

TEST(BigramModelTest, ReturnsEmptyStringForUnknownWord) {
    BigramModel model;
    model.train("hello world");
    EXPECT_EQ(model.predictNext("apple"), "");
}

TEST(LanguageModelTest, GenerateSentenceStopsWhenNoContinuationExists) {
    BigramModel model;
    model.train("hello world");

    EXPECT_EQ(model.generateSentence("world", 5), "world");
}

TEST(LanguageModelTest, GenerateSentenceBuildsDeterministicChainWhenOnlyOneChoiceExists) {
    BigramModel model;
    model.train("hello world there");

    EXPECT_EQ(model.generateSentence("hello", 3), "hello world there");
}
