#ifndef ORTCLS_H
#define ORTCLS_H

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <iterator>
#include <cstring>

#define MY_DLL extern "C" __declspec(dllexport)

MY_DLL void printInfo();
MY_DLL void run(const char* image_path, const char* onnx_path);

struct CLS_RES{
    short cls{-1};
    double confidence{-1};

    CLS_RES(short cls_, double confidence_):cls(cls_), confidence(confidence_) {}
};

/// @brief 加载模型文件 .onnx，初始化模型
/// @param model_pth onnx模型文件路径字符串指针
/// @param msg 消息字符数组，用于写入信息
/// @return 
MY_DLL void initModel(const char* model_pth, char* msg);

/// @brief 根据图片名+ROI 进行标签分类
/// @param image_pth 图片路径
/// @param roi 检查区域ROI 整型数组 [p1_x, p1_y, p2_x, p2_y]  ROI=nullptr时直接对全图推理
/// @param msg 消息字符数组，用于写入信息 
/// @return 返回分类标签
MY_DLL CLS_RES doInferenceByImgPth(const char* image_pth, const int* roi, char* msg);

/// @brief 根据 图像指针+图像尺寸 进行标签分类
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param compiled_model OpenVINO 模型指针 
/// @param msg 消息字符数组，用于写入信息 
/// @return 返回分类标签
MY_DLL CLS_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, char* msg);

MY_DLL void destroyModel();
CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, char* msg);
std::string getTimeNow();


#endif