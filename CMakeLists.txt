cmake_minimum_required(VERSION 3.24)
project(MyProject)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(bigram bigram.cpp)
#копирую датасет
file(COPY bigram.txt DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

#гугл тест
include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)
FetchContent_MakeAvailable(googletest)
enable_testing()

#тест
add_executable(run_tests test.cpp)
target_link_libraries(run_tests GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(run_tests)