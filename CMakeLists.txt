cmake_minimum_required(VERSION 3.24)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main
        nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.cpp
        nanoArkadiiGPT/ArkadiiGPT/nanoArkadiiGPT.h
        nanoArkadiiGPT/main.cpp
        nanoArkadiiGPT/config.h
        nanoArkadiiGPT/head/head.h
        nanoArkadiiGPT/head/head.cpp
        nanoArkadiiGPT/multihead_attention/multihead_attention.h
        nanoArkadiiGPT/multihead_attention/multihead_attention.cpp
        nanoArkadiiGPT/transformerblock/transformerblock.h
        nanoArkadiiGPT/transformerblock/tranformerblock.cpp
)

find_package(Torch REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)
target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} Qt6::Widgets)