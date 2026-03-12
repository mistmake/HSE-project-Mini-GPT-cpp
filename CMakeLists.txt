cmake_minimum_required(VERSION 3.24)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(Protobuf_DIR "/opt/homebrew/opt/protobuf@33/lib/cmake/protobuf")
find_package(Torch REQUIRED)
find_package(Qt6 REQUIRED COMPONENTS Widgets)

add_executable(main
        nanoArkadiiGPT/app/app.cpp
        nanoArkadiiGPT/app/mainwindow.h
)

target_link_libraries(main PRIVATE ${TORCH_LIBRARIES} Qt6::Widgets)