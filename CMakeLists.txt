cmake_minimum_required(VERSION 3.24)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_TESTS "Build GoogleTest tests" OFF)

#set(CMAKE_PREFIX_PATH
#        "/opt/homebrew/opt/pytorch/libexec/lib/python3.14/site-packages/torch"
#        "/opt/homebrew/opt/protobuf@33"
#        ${CMAKE_PREFIX_PATH}
#)

#set(Protobuf_DIR "/opt/homebrew/opt/protobuf@33/lib/cmake/protobuf" CACHE PATH "" FORCE)

find_package(Torch REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_library(arkadii_core
        nanoArkadiiGPT/config.h
        nanoArkadiiGPT/head/head.cpp
        nanoArkadiiGPT/head/head.h
        nanoArkadiiGPT/multihead_attention/multihead_attention.cpp
        nanoArkadiiGPT/multihead_attention/multihead_attention.h
        nanoArkadiiGPT/transformerblock/tranformerblock.cpp
        nanoArkadiiGPT/transformerblock/transformerblock.h
        nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.cpp
        nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.h
        nanoArkadiiGPT/config.cpp


)

add_library(milana_bigram
        bigram/bigram.cpp
        bigram/bigrammodel/bigrammodel.cpp
        bigram/languagemodel/languagemodel.cpp
        bigram/dataloader/dataloader.cpp
        bigram/dataloader/dataloader.h
        bigram/languagemodel/languagemodel.h
        bigram/bigrammodel/bigrammodel.h
)

add_library(simeon_dataset
        dataset/tokenizer.cpp
        dataset/TokenizerConfig.cpp
        dataset/TokenizerConfig.h
        dataset/processing/funcs.cpp
        dataset/utils/utils.cpp
        dataset/utils/utils.h
        dataset/processing/funcs.h
)

target_include_directories(arkadii_core
        PUBLIC
        ${CMAKE_SOURCE_DIR}/nanoArkadiiGPT
        ${TORCH_INCLUDE_DIRS}
)

target_compile_options(arkadii_core PUBLIC ${TORCH_CXX_FLAGS})
target_link_libraries(arkadii_core PUBLIC ${TORCH_LIBRARIES})

add_executable(main
        nanoArkadiiGPT/app/app.cpp
        nanoArkadiiGPT/app/mainwindow.cpp
        nanoArkadiiGPT/app/mainwindow.h
        nanoArkadiiGPT/app/mainwindow.ui
)
add_executable(bigram
bigram/bigram.cpp)

set_target_properties(main PROPERTIES
        AUTOMOC ON
        AUTOUIC ON
        AUTORCC ON
        OUTPUT_NAME app
)

target_link_libraries(main PRIVATE arkadii_core Qt6::Widgets)
target_link_libraries(bigram PRIVATE milana_bigram)

if(ENABLE_TESTS)
    include(FetchContent)

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    add_executable(arkadii_tests
            nanoArkadiiGPT/tests/test_main.cpp
            nanoArkadiiGPT/tests/model_logic_test.cpp
    )

    set_target_properties(arkadii_tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    target_include_directories(arkadii_tests PRIVATE ${CMAKE_SOURCE_DIR}/nanoArkadiiGPT)
    target_link_libraries(arkadii_tests PRIVATE arkadii_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(arkadii_tests)
endif()
