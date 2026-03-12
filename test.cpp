#define HIDE_MAIN
#include "bigram.cpp"
#include <gtest/gtest.h>

TEST(DataLoaderTest, ReturnsNulloptWhenFileNotFound) {
    DataLoader loader;
    auto result = loader.loadText("fake_missing_file.txt");
    EXPECT_FALSE(result.has_value());
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