set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

if(OpenCV_FOUND)
    message(STATUS "found opencv")
else()
    message(STATUS "not found opencv")
endif()

message(STATUS "Now Project is OpenCV Demo")
add_executable(opencv_test pass_mat.cxx utils.cxx)
target_include_directories(opencv_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${OpenCV_LIBS})
target_link_libraries(opencv_test PUBLIC ${OpenCV_LIBS})
