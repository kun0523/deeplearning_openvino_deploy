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

struct InferInfo{
    int id_;
    size_t roi_left{0};
    size_t roi_top{0};
    double scale_ratio{1};
    size_t left_padding{0};
    size_t top_padding{0};
    ov::InferRequest request;

    InferInfo(int _id_, ov::InferRequest req, size_t roi_lt=0, size_t roi_tp=0, double scale_=1, size_t lpad_=0, size_t tpad_=0):id_(_id_), roi_left(roi_lt), roi_top(roi_tp), scale_ratio(scale_), left_padding(lpad_), top_padding(tpad_), request(req){}
};


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
/// @param model_type 指定使用的模型版本 v8 v10 v11
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 DET_RES 数组指针 包含多个检测结果
MY_DLL DET_RES* doInferenceByImgPth(const char* img_pth, void* model_ptr, const int* roi, const float score_threshold, const short model_type, size_t& det_num, char* msg);


/// @brief 根据 图像指针+图像尺寸 进行目标检测
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param model_ptr OpenVINO 模型指针
/// @param score_threshold 置信度阈值 低于阈值的检测框不返回
/// @param model_type 指定使用的模型版本 v8 v10 v11
/// @param det_num 返回检测到的目标数量
/// @param msg 消息字符数组，用于写入信息 默认数组长度1024
/// @return 返回 DET_RES 数组指针 包含多个检测结果
MY_DLL DET_RES* doInferenceBy3chImg(uchar* image_arr, const int height, const int width, void* model_ptr, const float score_threshold, const short model_type, size_t& det_num, char* msg);


/// TODO: 速度差不多，但是精度有差异。。。先看一下fastdeploy怎么做的，学一学再改
/// @brief 传入多个相同尺寸的小图像指针数组，返回各个小图像的检测结果  
/// @param imgs_ptr_arr  图像指针数组 
/// @param imgs_num  数组中有几个图像
/// @param height 图像高度
/// @param width  图像宽度
/// @param model_ptr 模型指针
/// @param score_threshold 过滤的分数阈值
/// @param model_type 指定使用的模型版本 v8 v10 v11 
/// @param msg Log信息
/// @return  返回 vector<vector<DET_RES>>* 指针
void* doInferenceBy3chImgPatches(uchar** imgs_ptr_arr, const size_t imgs_num, const int height, const int width, 
                                    void* model_ptr, const float score_threshold, const short model_type, char* msg);


void warmUp(void* model_ptr, std::stringstream& msg_ss_);
DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, const float score_threshold, const short model_type, size_t& det_num, char* msg);
char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img, double& scale_ratio, int& left_padding, int& top_padding);
char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor);
