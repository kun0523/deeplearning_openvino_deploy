cmake_minimum_required(VERSION 3.26)
project("MyDeploy")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)


# add_subdirectory(src/paddle_ocr)
# add_executable(PaddleDemo app/paddle_infer.cpp)
# target_link_libraries(PaddleDemo PUBLIC paddle_ocr_infer)

# add_subdirectory(src/paddle_det)
# add_executable(PaddleDemo app/paddle_infer.cpp)
# target_link_libraries(PaddleDemo PUBLIC paddle_det_infer)

####################################### 视觉任务 ####################################################

# 使用OnnxRuntime推理框架执行 分类 检测 分割任务
option(USE_ORT_CLS "use onnxruntime do classification" ON)
if(USE_ORT_CLS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ort_bin)
    add_subdirectory(src/onnxruntime/cls_infer)
    add_executable(ORTDemo_cls app/ort_test/onnx_cls_infer.cpp)
    target_link_libraries(ORTDemo_cls PUBLIC ort_cls)
endif()

option(USE_ORT_DET "use onnxruntime do detection" ON)
if(USE_ORT_DET)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ort_bin)
    add_subdirectory(src/onnxruntime/det_infer)
    add_executable(ORTDemo_det app/ort_test/onnx_det_infer.cpp)
    target_link_libraries(ORTDemo_det PUBLIC ort_det)
endif()

option(USE_ORT_SEG "use onnxruntime do segmentation" ON)
if(USE_ORT_SEG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/ort_bin)
    add_subdirectory(src/onnxruntime/seg_infer)
    add_executable(ORTDemo_seg app/ort_test/onnx_seg_infer.cpp)
    target_link_libraries(ORTDemo_seg PUBLIC ort_seg)
endif()


# 使用OpenVINO推理框架执行 分类 检测 分割任务
option(USE_OPENVINO_CLS "use openvino do classification" ON)
if(USE_OPENVINO_CLS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/openvino_bin)
    add_subdirectory(src/openvino/cls_infer)
    add_executable(OpenVINODemo_cls app/openvino_test/openvino_cls_infer.cpp)
    target_link_libraries(OpenVINODemo_cls PUBLIC openvino_cls)
endif()

option(USE_OPENVINO_DET "use openvino do detection" ON)
if(USE_OPENVINO_DET)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/openvino_bin)
    add_subdirectory(src/openvino/det_infer)
    add_executable(OpenVINODemo_det app/openvino_test/openvino_det_infer.cxx)
    target_link_libraries(OpenVINODemo_det PUBLIC openvino_det)
endif()

option(USE_OPENVINO_SEG "use openvino do segmentation" ON)
if(USE_OPENVINO_SEG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/openvino_bin)
    add_subdirectory(src/openvino/seg_infer)
    add_executable(OpenVINODemo_seg app/openvino_test/openvino_seg_infer.cxx)
    target_link_libraries(OpenVINODemo_seg PUBLIC openvino_seg)
endif()


# 使用TensorRT推理框架执行 分类 检测 分割任务
option(USE_TENSORRT_CLS "use TensorRT do classification" ON)
if(USE_TENSORRT_CLS)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/trt_bin)
    add_subdirectory(src/tensorrt/cls_infer)
    add_executable(TensorRTDemo_cls app/trt_test/trt_cls_infer.cxx)
    target_link_libraries(TensorRTDemo_cls PUBLIC trt_cls)
endif()

option(USE_TENSORRT_DET "use TensorRT do detection" ON)
if(USE_TENSORRT_DET)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/trt_bin)
    add_subdirectory(src/tensorrt/det_infer)
    add_executable(TensorRTDemo_det app/trt_test/trt_det_infer.cxx)
    target_link_libraries(TensorRTDemo_det PUBLIC trt_det)
endif()

option(USE_TENSORRT_SEG "use TensorRT do segmentation" ON)
if(USE_TENSORRT_SEG)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/trt_bin)
    add_subdirectory(src/tensorrt/seg_infer)
    add_executable(TensorRTDemo_seg app/trt_test/trt_seg_infer.cxx)
    target_link_libraries(TensorRTDemo_seg PUBLIC trt_seg)
endif()

# 使用FastDeploy推理框架执行 分类 检测 分割 OCR任务


####################################### 临时测试 ####################################################

# 遍历文件夹复制文件到指定路径
option(COPY_FILES "Compile Copy Files Proj" OFF)
if(COPY_FILES)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/test_proj)
    add_executable(CopyFiles app/copyfiles/main.cxx)
endif()


# 测试OpenCV 特性
# TODO OpenCV CUDA 
option(OPENCV_TEST "Compile OpenCV Test Proj" OFF)
if(OPENCV_TEST)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/test_proj)
    add_subdirectory(app/opencv_test)
endif()
