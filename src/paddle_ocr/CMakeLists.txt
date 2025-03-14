project(paddle_ocr_infer)

set(FASTDEPLOY_INSTALL_DIR "E:\\le_fastdeploy\\fastdeploy-win-x64-1.0.7")
include(${FASTDEPLOY_INSTALL_DIR}/FastDeploy.cmake)

message(STATUS "Now Project is ${PROJECT_NAME}")
add_library(${PROJECT_NAME} SHARED inference.cxx inference.h)
# add_executable(${PROJECT_NAME} inference.cxx inference.h)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(${PROJECT_NAME} PUBLIC ${FASTDEPLOY_INCS})
# target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${FASTDEPLOY_LIBS})