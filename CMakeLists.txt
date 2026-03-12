cmake_minimum_required(VERSION 3.24)
project(BigramProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ==========================================
# 1. НАСТРОЙКА GOOGLE TEST
# ==========================================
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

# ==========================================
# 2. ОСНОВНАЯ ПРОГРАММА
# ==========================================
# Собирает файл bigram.cpp и запускает его обычный main()
add_executable(bigram_run bigram.cpp)

# ==========================================
# 3. ТЕСТЫ
# ==========================================
# Собирает только test.cpp (а он сам внутри себя подтянет bigram.cpp)
add_executable(run_tests test.cpp)
target_link_libraries(run_tests PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)