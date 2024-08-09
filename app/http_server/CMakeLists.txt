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