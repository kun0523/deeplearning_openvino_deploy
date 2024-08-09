project(cls_infer_lib)

set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH E:\\cpp_packages\\w_openvino_toolkit_windows_2024.2.0.15519.5c0f38f83f6_x86_64\\runtime\\cmake)
find_package(OpenVINO REQUIRED)

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

message(STATUS "Now Project is ${PROJECT_NAME}")
add_library(${PROJECT_NAME} SHARED inference.cxx inference.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC openvino::runtime)