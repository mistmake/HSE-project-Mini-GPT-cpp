cmake_minimum_required(VERSION 3.24)

project(HSEMiniGPT LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(BUILD_APP "Build the Qt/Torch application" ON)
option(BUILD_BIGRAM "Build the bigram module" ON)
option(BUILD_TOKENIZER "Build the tokenizer module" ON)
option(ENABLE_TESTS "Build GoogleTest tests" ON)

function(configure_cpp_target target_name)
    target_compile_features(${target_name} PUBLIC cxx_std_23)
endfunction()

function(use_project_root_as_working_directory target_name)
    set_target_properties(${target_name} PROPERTIES
            VS_DEBUGGER_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
            XCODE_GENERATE_SCHEME YES
            XCODE_SCHEME_WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    )
endfunction()

if(BUILD_APP OR ENABLE_TESTS)
    find_package(Torch REQUIRED)

    add_library(arkadii_core
            nanoArkadiiGPT/config.cpp
            nanoArkadiiGPT/config.h
            nanoArkadiiGPT/head/head.cpp
            nanoArkadiiGPT/head/head.h
            nanoArkadiiGPT/multihead_attention/multihead_attention.cpp
            nanoArkadiiGPT/multihead_attention/multihead_attention.h
            nanoArkadiiGPT/transformerblock/tranformerblock.cpp
            nanoArkadiiGPT/transformerblock/transformerblock.h
            nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.cpp
            nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.h
    )

    configure_cpp_target(arkadii_core)

    target_include_directories(arkadii_core
            PUBLIC
            "${PROJECT_SOURCE_DIR}/nanoArkadiiGPT"
            ${TORCH_INCLUDE_DIRS}
    )

    target_compile_options(arkadii_core PUBLIC ${TORCH_CXX_FLAGS})
    target_link_libraries(arkadii_core PUBLIC ${TORCH_LIBRARIES})
endif()

if(BUILD_BIGRAM)
    add_library(bigram_core
            bigram/bigrammodel/bigrammodel.cpp
            bigram/bigrammodel/bigrammodel.h
            bigram/languagemodel/languagemodel.cpp
            bigram/languagemodel/languagemodel.h
            bigram/dataloader/dataloader.cpp
            bigram/dataloader/dataloader.h
    )

    configure_cpp_target(bigram_core)

    target_include_directories(bigram_core
            PUBLIC
            "${PROJECT_SOURCE_DIR}/bigram"
    )

    add_executable(bigram
            bigram/bigram.cpp
    )

    configure_cpp_target(bigram)
    target_link_libraries(bigram PRIVATE bigram_core)
    use_project_root_as_working_directory(bigram)
endif()

if(BUILD_TOKENIZER)
    add_library(tokenizer_core
            tokenizer/config/TokenizerConfig.cpp
            tokenizer/config/TokenizerConfig.h
            tokenizer/processing/funcs.cpp
            tokenizer/processing/funcs.h
            tokenizer/utils/utils.cpp
            tokenizer/utils/utils.h
    )

    configure_cpp_target(tokenizer_core)

    target_include_directories(tokenizer_core
            PUBLIC
            "${PROJECT_SOURCE_DIR}/tokenizer"
    )

    add_executable(tokenizer
            tokenizer/tokenizer.cpp
    )

    configure_cpp_target(tokenizer)
    target_link_libraries(tokenizer PRIVATE tokenizer_core)
    use_project_root_as_working_directory(tokenizer)
endif()

if(BUILD_APP)
    find_package(Qt6 REQUIRED COMPONENTS Widgets)

    add_executable(main
            nanoArkadiiGPT/app/app.cpp
            nanoArkadiiGPT/app/mainwindow.cpp
            nanoArkadiiGPT/app/mainwindow.h
            nanoArkadiiGPT/app/mainwindow.ui
    )

    configure_cpp_target(main)

    set_target_properties(main PROPERTIES
            AUTOMOC ON
            AUTOUIC ON
            AUTORCC ON
            OUTPUT_NAME app
    )

    target_link_libraries(main PRIVATE arkadii_core Qt6::Widgets)
    use_project_root_as_working_directory(main)
endif()

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

    configure_cpp_target(arkadii_tests)

    set_target_properties(arkadii_tests PROPERTIES
            AUTOMOC OFF
            AUTOUIC OFF
            AUTORCC OFF
    )

    target_include_directories(arkadii_tests PRIVATE "${PROJECT_SOURCE_DIR}/nanoArkadiiGPT")
    target_link_libraries(arkadii_tests PRIVATE arkadii_core GTest::gtest)

    include(GoogleTest)
    gtest_discover_tests(arkadii_tests)

    use_project_root_as_working_directory(arkadii_tests)

    if(BUILD_BIGRAM)
        add_executable(bigram_tests
                bigram/test.cpp
        )

        configure_cpp_target(bigram_tests)

        target_include_directories(bigram_tests PRIVATE "${PROJECT_SOURCE_DIR}/bigram")
        target_link_libraries(bigram_tests PRIVATE bigram_core GTest::gtest_main)

        gtest_discover_tests(bigram_tests)
        use_project_root_as_working_directory(bigram_tests)
    endif()

    if(BUILD_TOKENIZER)
        add_executable(tokenizer_tests
                tokenizer/tests/tokenizer_logic_test.cpp
        )

        configure_cpp_target(tokenizer_tests)

        target_include_directories(tokenizer_tests PRIVATE "${PROJECT_SOURCE_DIR}/tokenizer")
        target_link_libraries(tokenizer_tests PRIVATE tokenizer_core GTest::gtest_main)

        gtest_discover_tests(tokenizer_tests)
        use_project_root_as_working_directory(tokenizer_tests)
    endif()
endif()
