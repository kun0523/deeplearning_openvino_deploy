project(cls_infer_sklearn_lib)
message(STATUS "Now Project is ${PROJECT_NAME}")

set(ONNX_RUNTIME_ROOT "E:\\cpp_packages\\onnxruntime-win-x64-1.19.2")
find_library(ONNX_RUNTIME_LIBRARY 
             NAMES onnxruntime 
             PATHS ${ONNX_RUNTIME_ROOT}/lib
             )

add_library(${PROJECT_NAME} SHARED inference.cxx inference.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_ROOT}/include)
set(SPDLOG_DIR "E:\\cpp_packages\\spdlog-1.x")
target_include_directories(${PROJECT_NAME} PUBLIC ${SPDLOG_DIR}/include)

target_link_libraries(${PROJECT_NAME} PUBLIC ${ONNX_RUNTIME_LIBRARY})
