#include <cstdio>
#include <iostream>
#include "cpp-httplib/httplib.h"
#include <curl/curl.h>
#include <opencv2/opencv.hpp>
#include <chrono>

using namespace httplib;

std::string dump_headers(const Headers &headers) {
  std::string s;
  char buf[BUFSIZ];

  for (auto it = headers.begin(); it != headers.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%s: %s\n", x.first.c_str(), x.second.c_str());
    s += buf;
  }

  return s;
}

std::string log(const Request &req, const Response &res) {
  std::string s;
  char buf[BUFSIZ];

  s += "================================\n";

  snprintf(buf, sizeof(buf), "%s %s %s", req.method.c_str(),
           req.version.c_str(), req.path.c_str());
  s += buf;

  std::string query;
  for (auto it = req.params.begin(); it != req.params.end(); ++it) {
    const auto &x = *it;
    snprintf(buf, sizeof(buf), "%c%s=%s",
             (it == req.params.begin()) ? '?' : '&', x.first.c_str(),
             x.second.c_str());
    query += buf;
  }
  snprintf(buf, sizeof(buf), "%s\n", query.c_str());
  s += buf;

  s += dump_headers(req.headers);

  s += "--------------------------------\n";

  snprintf(buf, sizeof(buf), "%d %s\n", res.status, res.version.c_str());
  s += buf;
  s += dump_headers(res.headers);
  s += "\n";

  if (!res.body.empty()) { s += res.body; }

  s += "\n";

  return s;
}

//curl writefunction to be passed as a parameter
// we can't ever expect to get the whole image in one piece,
// every router / hub is entitled to fragment it into parts
// (like 1-8k at a time),
// so insert the part at the end of our stream.
size_t write_data(char *ptr, size_t size, size_t nmemb, void *userdata)
{
    std::vector<uchar> *stream = (std::vector<uchar>*)userdata;
    size_t count = size * nmemb;
    stream->insert(stream->end(), ptr, ptr + count);
    return count;
}

//function to retrieve the image as cv::Mat data type
cv::Mat curlImg(const char *img_url, int timeout=10)
{
    std::vector<uchar> stream;
    CURL *curl = curl_easy_init();
    curl_easy_setopt(curl, CURLOPT_URL, img_url); //the img url
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_data); // pass the writefunction
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &stream); // pass the stream ptr to the writefunction
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, timeout); // timeout if curl_easy hangs, 
    CURLcode res = curl_easy_perform(curl); // start curl
    curl_easy_cleanup(curl); // cleanup
    return cv::imdecode(stream, -1); // 'keep-as-is'
}


/*
TODO：构建HTTP Server，接收外部请求，根据请求的形式，做分类、检测、分割等任务
*/

int main(){
    httplib::Server svr;
    svr.Get("/hi", [](const Request &, Response &res){
        res.set_content("Hello You Beautiful People!", "text/plain");

    });

    svr.Get("/classify", [](const Request &, Response &res){
        res.set_content("Do Classify ...", "text/plain");
    });

    svr.Get("/detect", [](const Request &, Response &res){
        res.set_content("Do Detect ...", "text/plain");
    });

    svr.Get("/segment", [](const Request &, Response &res){
        res.set_content("Do Segment ...", "text/plain");
    });

    svr.set_logger([](const Request &req, const Response &res){
        if(req.params.size()<1){
            std::cout << "No Params" << std::endl;
        }
        for(auto item : req.params){
            std::cout << "request params: " << item.first << ":" << item.second << std::endl;
            auto read_img_start = std::chrono::high_resolution_clock::now();
            cv::Mat image = curlImg(item.second.c_str());
            auto read_img_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> spend = read_img_end - read_img_start;
            std::cout << "Read Image cost: " << spend.count() << "ms" << std::endl;
            if(image.empty()){
                std::cout << "Image is Empty!" << std::endl;
            }else{
                cv::namedWindow("Show", cv::WINDOW_NORMAL);
                cv::imshow("Show", image);
                cv::waitKey(0);
            }
        }
        std::cout << "response body: " << res.body << std::endl;

        std::cout << log(req, res) << std::endl;

        // 对比读取本地图像
        {            
            std::string img_pth = R"(E:\test_images\_432_1436_202406291446324_5.jpg)";  // 12K image
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat image = cv::imread(img_pth);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> spend = end - start;
            std::cout << "Read Image " << img_pth << " cost:\n\t" << spend.count() << std::endl; 
        }

        {            
            std::string img_pth = R"(E:\test_images\20240805_00001_B12_QRCODE=A743216312_14_17_C3_FZ_A2FZ2SD7ANOBD081_A2FZ2S47DJIDC124.jpg)";  // 1560K image
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat image = cv::imread(img_pth);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> spend = end - start;
            std::cout << "Read Image " << img_pth << " cost:\n\t" << spend.count() << std::endl; 
        }

        {            
            std::string img_pth = R"(E:\test_images\20240806_00001_B12_QRCODE=A743219099_2_17_C3_FW_A2FW1S47GG9DF059_A2FW1S47EDGCB064.jpg)";  // 3255K
            auto start = std::chrono::high_resolution_clock::now();
            cv::Mat image = cv::imread(img_pth);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> spend = end - start;
            std::cout << "Read Image " << img_pth << " cost:\n\t" << spend.count() << std::endl; 
        }

    });

    // read         Remote      Local
    // image 12K    28ms        0.64ms
    // image 1560K  260ms       129ms
    // image 3255K  427ms       145ms

    // svr.listen("127.0.0.1", 8080);
    svr.listen("192.168.0.51", 8080);
    return 0;
}