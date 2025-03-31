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
#include <cmath>

using std::endl;
using std::string;
using std::vector;

#define MY_DLL extern "C" __declspec(dllexport)

MY_DLL void printInfo();
MY_DLL void run(const char* image_path, const char* onnx_path);

Ort::Env* ENV_PTR=nullptr;

struct DET_RES{
    int tl_x{-1};
    int tl_y{-1};
    int br_x{-1};
    int br_y{-1};
    int cls{-1};
    double confidence{-1};
    DET_RES()=default;
    DET_RES(const cv::Rect2d rec_, int cls_id_, double conf_):tl_x(static_cast<int>(rec_.x)), 
                                                            tl_y(static_cast<int>(rec_.y)),    
                                                            br_x(static_cast<int>(rec_.x+rec_.width)),
                                                            br_y(static_cast<int>(rec_.y+rec_.height)), 
                                                            cls(cls_id_), confidence(conf_){}
    DET_RES(const cv::Rect2i rec_, int cls_id_, double conf_):tl_x(rec_.x), 
                                                            tl_y(rec_.y), 
                                                            br_x(rec_.x+rec_.width), 
                                                            br_y(rec_.y+rec_.height), 
                                                            cls(cls_id_), confidence(conf_){}
    int get_area();
    std::string get_info();
};


/// @brief 输入onnx模型文件路径 返回初始化完成的模型指针
/// @param onnx_pth onnx模型文件路径字符串指针
/// @param msg 消息字符数组，用于写入Log信息，默认数组长度1024
/// @return 返回初始化后的模型指针
MY_DLL void* initModel(const char* onnx_pth, char* msg);


/// @brief 根据图片名+ROI 进行目标检测
/// @param img_pth 图片路径
/// @param model_ptr OpenVINO 模型指针
/// @param roi 检查区域ROI 整型数组 [p1_x, p1_y, p2_x, p2_y]  ROI=nullptr时直接对全图推理
/// @param score_threshold 置信度阈值 低于阈值的检测框不返回
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 DET_RES 数组指针 包含多个检测结果
MY_DLL DET_RES* doInferenceByImgPth(const char* img_pth, void* model_ptr, const int* roi, const float score_threshold, size_t& det_num, char* msg);


/// @brief 根据 图像指针+图像尺寸 进行目标检测
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param model_ptr OpenVINO 模型指针
/// @param score_threshold 置信度阈值 低于阈值的检测框不返回
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 DET_RES 数组指针 包含多个检测结果
MY_DLL DET_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, const float score_threshold, size_t& det_num, char* msg);


MY_DLL void destroyModel(void* model_ptr);
DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* model_ptr, const float score_threshold, size_t& det_num, char* msg);
std::string getTimeNow();

void* preProcess(const Ort::Session* infer_session, const cv::Mat& org_img, cv::Mat& boarded_img);
void postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, std::vector<DET_RES>& out_vec);


#endif