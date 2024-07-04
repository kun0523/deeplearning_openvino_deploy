cmake_minimum_required(VERSION 3.10)
project("openvino_deploy")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(Cls_Infer_Dir ${CMAKE_SOURCE_DIR}/src/cls_infer)
set(Det_Infer_Dir ${CMAKE_SOURCE_DIR}/src/det_infer)

add_subdirectory(src)