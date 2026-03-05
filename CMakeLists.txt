cmake_minimum_required(VERSION 3.24)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_executable(main nanoArkadiiGPT/nanoArkadiiGPT.cpp)

target_link_libraries(main PRIVATE ${TORCH_LIBRARIES})