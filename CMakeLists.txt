cmake_minimum_required(VERSION 3.26)
project("openvino_deploy")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

set(Sklearn_Infer_Dir ${CMAKE_SOURCE_DIR}/src/sklearn_infer)
set(Cls_Infer_Dir ${CMAKE_SOURCE_DIR}/src/cls_infer)
set(Det_Infer_Dir ${CMAKE_SOURCE_DIR}/src/det_infer)
set(Det_Infer_Opvno_Dir ${CMAKE_SOURCE_DIR}/src/det_infer_opvno)

option(COMPILE_PADDLE_CLS_LIB "Whether or not to compile paddlecls library" OFF)
option(COMPILE_PADDLE_DET_LIB "Whether or not to compile paddledet library" OFF)
option(COMPILE_PADDLE_SEG_LIB "Whether or not to compile paddleseg library" OFF)
option(COMPILE_PADDLE_OCR_LIB "Whether or not to compile paddleocr library" OFF)

option(COMPILE_SKLEARN_LIB "Whether or not to compile sklearn library" OFF)
option(COMPILE_CLS_LIB "Whether or not to compile classify library" OFF)
option(COMPILE_DET_LIB "Whether or not to compile detect library" OFF)
option(COMPILE_DET_OPVNO_LIB "Whether or not to compile detect library" OFF)
option(COMPILE_SEG_LIB "Whether or not to compile segment library" OFF)
option(COMPILE_OCR_LIB "Whether or not to compile ocr library" OFF)
option(COMPILE_HTTP_SERVER "Whether or not to compile http server" OFF)

# add_subdirectory(src)
# add_subdirectory(app)

# add_subdirectory(src/paddle_ocr)
# add_executable(PaddleDemo app/paddle_infer.cpp)
# target_link_libraries(PaddleDemo PUBLIC paddle_ocr_infer)

add_subdirectory(src/paddle_det)
add_executable(PaddleDemo app/paddle_infer.cpp)
target_link_libraries(PaddleDemo PUBLIC paddle_det_infer)