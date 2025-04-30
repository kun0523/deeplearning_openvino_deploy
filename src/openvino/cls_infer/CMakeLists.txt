set(CMAKE_PREFIX_PATH E:\\cpp_packages\\w_openvino_toolkit_windows_2024.2.0.15519.5c0f38f83f6_x86_64\\runtime\\cmake)
find_package(OpenVINO REQUIRED)

set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "found opencv")
else()
    message(STATUS "not found opencv")
endif()

if(OpenVINO_FOUND)
    message(STATUS "found openvino")
else()
    message(STATUS "not found openvino")
endif()

message(STATUS "Now Project is openvino_cls")
add_library(openvino_cls SHARED inference.cxx)
target_compile_definitions(openvino_cls PRIVATE "DEBUG_OPV")  # 如果定义DEBUG 应影响opencv的链接！！！！导致链接不到！！

target_include_directories(openvino_cls PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(openvino_cls PUBLIC ${OpenCV_LIBS} openvino::runtime)
