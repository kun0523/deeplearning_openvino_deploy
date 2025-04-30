set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

set(TRT_DIR E:\\cpp_packages\\TensorRT-8.6.1.6)
set(NVTOOLKIT_DIR C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8)
enable_language(CUDA)  # 必须有
find_package(CUDA REQUIRED)

find_library(TENSORRT_LIB nvinfer PATHS ${TRT_DIR}/lib)
find_library(NVONNXPARSER_LIB nvonnxparser PATHS ${TRT_DIR}/lib)

if(OpenCV_FOUND)
    message(STATUS "found opencv")
else()
    message(STATUS "not found opencv")
endif()

message(STATUS "Now Project is trt_det")
add_library(trt_det SHARED inference.cxx)

target_compile_definitions(trt_det PRIVATE "DEBUG_TRT") 
target_include_directories(trt_det PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} 
                                          ${TRT_DIR}/include 
                                          )

target_link_libraries(trt_det PUBLIC ${OpenCV_LIBS} ${TENSORRT_LIB} ${NVONNXPARSER_LIB} cudart)
