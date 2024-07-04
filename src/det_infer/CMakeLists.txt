project(det_infer_lib)

set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH E:\\cpp_packages\\w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64\\w_openvino_toolkit_windows_2023.2.0.13089.cfd42bd2cb0_x86_64\\runtime\\cmake)
find_package(OpenVINO REQUIRED)

add_library(${PROJECT_NAME} SHARED inference.cxx)
# target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC openvino::runtime)