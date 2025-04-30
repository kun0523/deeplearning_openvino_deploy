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

message(STATUS "Now Project is OpenVINO Segmentation")
add_library(openvino_seg SHARED inference.cxx)
target_compile_definitions(openvino_seg PRIVATE "DEBUG_OPV")  # 如果定义DEBUG 应影响opencv的链接！！！！导致链接不到！！

target_include_directories(openvino_seg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(openvino_seg PUBLIC ${OpenCV_LIBS} openvino::runtime)
