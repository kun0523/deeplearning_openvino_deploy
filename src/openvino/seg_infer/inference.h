#ifndef OPENVINOSEG_H
#define OPENVINOSEG_H

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <string>

#define MY_DLL extern "C" __declspec(dllexport)

using std::string;
using std::cin;
using std::cout;
using std::endl;
using std::vector;


struct SEG_RES{
    int tl_x{-1};
    int tl_y{-1};
    int br_x{-1};
    int br_y{-1};
    int cls{-1};
    double confidence{-1};
    int mask_h{};
    int mask_w{};
    int mask_type{};
    uchar* mask_data{nullptr};   // 仅保留 检测框内的 mask

    SEG_RES()=default;
    SEG_RES(const cv::Rect2d rec_, int cls_id_, double conf_):tl_x(static_cast<int>(rec_.x)), 
                                                            tl_y(static_cast<int>(rec_.y)),    
                                                            br_x(static_cast<int>(rec_.x+rec_.width)),
                                                            br_y(static_cast<int>(rec_.y+rec_.height)), 
                                                            cls(cls_id_), confidence(conf_){}
    SEG_RES(const cv::Rect2i rec_, int cls_id_, double conf_):tl_x(rec_.x), 
                                                            tl_y(rec_.y), 
                                                            br_x(rec_.x+rec_.width), 
                                                            br_y(rec_.y+rec_.height), 
                                                            cls(cls_id_), confidence(conf_){}
    int get_area();
    std::string get_info();

    ~SEG_RES();
};


/// @brief 加载模型文件 .onnx，初始化模型
/// @param model_pth onnx模型文件路径字符串指针
/// @param msg 消息字符数组，用于写入信息
/// @return 
MY_DLL void initModel(const char* model_pth, char* msg);


/// @brief 根据图片名+ROI 进行目标检测
/// @param img_pth 图片路径
/// @param roi 检查区域ROI 整型数组 [p1_x, p1_y, p2_x, p2_y]  ROI=nullptr时直接对全图推理
/// @param score_threshold 置信度阈值 低于阈值的检测框不返回
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 SEG_RES 数组指针 包含多个检测结果
MY_DLL SEG_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const float score_threshold, int& det_num, char* msg);


/// @brief 根据 图像指针+图像尺寸 进行目标检测
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param score_threshold 置信度阈值 低于阈值的检测框不返回
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 SEG_RES 数组指针 包含多个检测结果
MY_DLL SEG_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const float score_threshold, int& det_num, char* msg);


/// @brief 销毁内存中的模型，回收资源
/// @return 
MY_DLL void destroyModel();


void warmUp();
SEG_RES* doInferenceByImgMat(const cv::Mat& img_mat, const float score_threshold, int& det_num, char* msg);
void preProcess(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img);
SEG_RES* postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num);

#endif