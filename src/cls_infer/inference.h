#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>

using std::string;
using std::cin;
using std::cout;
using std::endl;
using std::vector;

#define MY_DLL extern "C" __declspec(dllexport)

struct CLS_RES{
    short cls{-1};
    double confidence{-1};

    CLS_RES(short cls_, double confidence_):cls(cls_), confidence(confidence_) {}
};

/// @brief 输入onnx模型文件路径 返回初始化完成的模型指针
/// @param onnx_pth onnx模型文件路径字符串指针
/// @param msg 消息字符数组，用于写入信息
/// @param msg_len 模型消息长度1024
/// @return 返回初始化后的模型指针
MY_DLL void* initModel(const char* onnx_pth, char* msg, size_t msg_len=1024);

/// @brief 根据图片名+ROI 进行标签分类
/// @param image_pth 图片路径
/// @param compiled_model OpenVINO 模型指针
/// @param roi 检查区域ROI 整型数组 [p1_x, p1_y, p2_x, p2_y]  ROI=nullptr时直接对全图推理
/// @param msg 消息字符数组，用于写入信息 
/// @param msg_len 指定返回消息的长度 默认1024
/// @return 返回分类标签
MY_DLL CLS_RES doInferenceByImgPth(const char* image_pth, void* compiled_model, const int* roi, char* msg, size_t msg_len=1024);

/// @brief 根据 图像指针+图像尺寸 进行标签分类
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param compiled_model OpenVINO 模型指针 
/// @param msg 消息字符数组，用于写入信息 
/// @param msg_len 指定返回消息的长度 默认1024
/// @return 返回分类标签
MY_DLL CLS_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* compiled_model, char* msg, size_t msg_len=1024);

// TODO: 待实现
int doInferenceBatchImgs(const char* image_dir, int height, int width, void* compiled_model, const int* roi, const int roi_len, char* msg, size_t msg_len=1024);

void warmUp(void* compiled_model, char* msg, size_t msg_len=1024);
CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, char* msg, size_t msg_len=1024);
char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img);
char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor);
string getTimeNow();
