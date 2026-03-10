// Сначала говорим компилятору спрятать твой main
#define HIDE_MAIN

// Теперь вклеиваем твой файл целиком!
#include "bigram.cpp"
#include <gtest/gtest.h>

// ТЕСТ 1: Проверяем исключения
TEST(DataLoaderTest, ThrowsExceptionWhenFileNotFound) {
    DataLoader loader;
    EXPECT_THROW(loader.loadText("fake_missing_file.txt"), FileIOException);
}

// ТЕСТ 2: Проверяем предсказания
TEST(BigramModelTest, PredictsCorrectCharacter) {
    BigramModel model;
    model.train("ab ac ab");
    EXPECT_EQ(model.predictNext('a'), 'b');
}