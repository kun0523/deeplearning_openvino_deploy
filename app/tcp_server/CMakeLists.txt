cmake_minimum_required(VERSION 3.13)
project(client)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(client client.cpp)
target_link_libraries(client ws2_32)

add_executable(server server_2.cpp)
target_link_libraries(server ws2_32)
