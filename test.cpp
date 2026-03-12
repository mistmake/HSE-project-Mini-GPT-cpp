// Сначала говорим компилятору спрятать твой main
#define HIDE_MAIN

// Теперь вклеиваем твой файл целиком!
#include "bigram.cpp"
#include <gtest/gtest.h>

// ТЕСТ 1: Проверяем работу DataLoader
TEST(DataLoaderTest, ReturnsNulloptWhenFileNotFound) {
    DataLoader loader;
    auto result = loader.loadText("fake_missing_file.txt");

    // Поскольку файла нет, мы ожидаем, что result не имеет значения (пустая коробочка std::optional)
    EXPECT_FALSE(result.has_value());
}

// ТЕСТ 2: Проверяем предсказания (теперь на уровне СЛОВ)
TEST(BigramModelTest, PredictsCorrectWord) {
    BigramModel model;

    // Обучаем модель на тексте, где всё очевидно: после "hello" всегда идет "world"
    model.train("hello world hello world hello world");

    // Проверяем: если дать слово "hello", она обязана вернуть "world"
    EXPECT_EQ(model.predictNext("hello"), "world");
}

// ТЕСТ 3: Проверяем защиту от неизвестных слов
TEST(BigramModelTest, ReturnsEmptyStringForUnknownWord) {
    BigramModel model;
    model.train("hello world");

    // Модель никогда не видела слово "apple". Мы запрограммировали её возвращать пустую строку ("")
    EXPECT_EQ(model.predictNext("apple"), "");
}