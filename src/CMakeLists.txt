project(test_model)

# add_subdirectory(cls_infer)
add_subdirectory(det_infer)
# add_subdirectory(ocr_infer)

add_executable(${PROJECT_NAME} do_infer.cpp)
# target_include_directories(${PROJECT_NAME} PUBLIC ${Cls_Infer_Dir})
# target_link_libraries(${PROJECT_NAME} PUBLIC cls_infer_lib)
target_link_libraries(${PROJECT_NAME} PUBLIC det_infer_lib)
# target_link_libraries(${PROJECT_NAME} PUBLIC ocr_infer_lib)

set(CMAKE_PREFIX_PATH D:\\opencv\\opencv\\build)
find_package(OpenCV REQUIRED)

set(ZLIB_INCLUDE_DIR E:/cpp_packages/vcpkg-master/packages/zlib_x64-windows/include)
find_package(ZLIB REQUIRED)
set(CMAKE_PREFIX_PATH E:/cpp_packages/vcpkg-master/packages/curl_x64-windows/share/curl)
# set(CURL_INCLUDE_DIR E://cpp_packages\\vcpkg-master\\packages\\curl_x64-windows\\include)
message(============)
find_package(CURL REQUIRED)
message(------------)

message(>>> ${CURL_LIBRARY})
add_executable(HttpServer http_server/server.cxx)
# include_directories(E:/cpp_packages/vcpkg-master/packages/curl_x64-windows/include)

# target_link_libraries(HttpServer ${CURL_LIBRARY})
target_link_libraries(HttpServer PRIVATE CURL::libcurl)
target_link_libraries(HttpServer PUBLIC ${OpenCV_LIBS})