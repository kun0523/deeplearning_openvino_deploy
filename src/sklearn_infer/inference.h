#include <iostream>
#include <vector>
#include <codecvt>
#include <onnxruntime_cxx_api.h>
#include <math.h>
#include <string>
#include <sstream>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/daily_file_sink.h>

#define MY_DLL extern "C" __declspec(dllexport)

struct CLS_RES{
    bool isNg{0};
    double confidence{-1};

    CLS_RES():isNg(0), confidence(-1){}
    CLS_RES(bool is_ng, double conf):isNg(is_ng), confidence(conf){}
    void print(){std::cout << "isNG: " << isNg << " confidence: " << confidence << std::endl;}
};

struct Point{
    double x{};
    double y{};

    Point():x(0), y(0){}
    Point(double x_, double y_):x(x_), y(y_){}

    Point operator+(const Point& other) const{
        return Point(x+other.x, y+other.y);
    }

    Point operator-(const Point& other) const{
        return Point(x-other.x, y-other.y);
    }

    std::string show(){
        std::stringstream ss;
        ss << " (" << x << "," << y << ") "; 
        return ss.str();
    }
};


/// @brief 输入onnx模型文件路径 返回初始化完成的模型指针
/// @param onnx_pth onnx模型文件路径字符串指针
/// @return 返回初始化后的模型指针
MY_DLL void* initModel(const char* model_pth);


/// @brief 根据45度附近5个点 判断OK/NG 并返回置信度
/// @param ort_session onnx 模型指针
/// @param base_point 基准点
/// @param points 数组 45度附近的几个点
/// @param p_num  points个数
/// @return 返回分类结果
MY_DLL CLS_RES doInference(void* ort_session, Point base_point, Point* points, size_t p_num);

std::vector<float> calDistances(Point base_point, Point* points, size_t p_num);