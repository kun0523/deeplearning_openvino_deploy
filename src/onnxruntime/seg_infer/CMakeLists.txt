set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "found opencv")
else()
    message(STATUS "not found opencv")
endif()

set(ONNXRUNTIME_DIR E:/cpp_packages/onnxruntime-win-x64-1.19.2)

message(STATUS "Now Project is ONNXRUNTIME Segmentation Demo")
add_library(ort_seg SHARED inference.cxx)
target_compile_definitions(ort_seg PRIVATE "DEBUG_ORT")  # 如果定义DEBUG 应影响opencv的链接！！！！导致链接不到！！
target_include_directories(ort_seg PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${ONNXRUNTIME_DIR}/include)
target_link_libraries(ort_seg PUBLIC ${OpenCV_LIBS} ${ONNXRUNTIME_DIR}/lib/onnxruntime.lib)