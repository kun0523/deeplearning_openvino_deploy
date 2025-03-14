#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <string>

#include "fastdeploy/vision.h"
#include <opencv2/opencv.hpp>

using std::string;
using std::cin;
using std::cout;
using std::endl;
using std::vector;

#ifdef WIN32
const char sep = '\\';
#else
const char sep = '/';
#endif

#define MY_DLL extern "C" __declspec(dllexport)

struct CLS_RES{
    int cls{-1};
    double confidence{-1};

    CLS_RES(int cls_, double confidence_):cls(cls_), confidence(confidence_) {}
};

struct OCR_ITEM{
    int bbox[8]{};  // 四点框，可以绘制倾斜矩形框
    char words[100]{};  // 存中文字符会有问题吗？
};

struct OCR_RES{
    std::vector<OCR_ITEM> items;
};

enum BACKEND{
    PADDLE,
    ONNX,
    OPENVINO
};

fastdeploy::vision::ocr::DBDetector* det_model_ptr = nullptr;
fastdeploy::vision::ocr::Classifier* cls_model_ptr = nullptr;
fastdeploy::vision::ocr::Recognizer* rec_model_ptr = nullptr;
fastdeploy::pipeline::PPOCRv3* ppocr = nullptr;

MY_DLL void printInfo(char* ret_msg, size_t msg_len=1024);

/// @brief 输入pdmodel模型文件路径 返回初始化完成的模型指针
/// @param pdmodel_dir pdmodel模型文件路径字符串指针
/// @param msg 消息字符数组，用于写入信息
/// @param msg_len 模型消息长度1024
/// @return 返回初始化后的模型指针
// MY_DLL void* initModel(const char* det_model_dir, const char* cls_model_dir, const char* rec_model_dir, const char* rec_dict_file, short backend_type, char* msg, size_t msg_len=1024);
MY_DLL void initModel(const char* det_model_dir, const char* cls_model_dir, const char* rec_model_dir, const char* rec_dict_file, short backend_type, char* msg, size_t msg_len=1024);

MY_DLL void destroyModel();

/// @brief 根据图片名+ROI 进行标签分类
/// @param image_pth 图片路径
/// @param compiled_model OpenVINO 模型指针
/// @param roi 检查区域ROI 整型数组 [p1_x, p1_y, p2_x, p2_y]  ROI=nullptr时直接对全图推理
/// @param msg 消息字符数组，用于写入信息 
/// @param msg_len 指定返回消息的长度 默认1024
/// @return 返回分类标签
MY_DLL OCR_RES doInferenceByImgPth(const char* image_pth, const double conf, char* msg, size_t msg_len=1024);

/// @brief 根据 图像指针+图像尺寸 进行标签分类
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针
/// @param height 图像高度
/// @param width 图像宽度
/// @param compiled_model OpenVINO 模型指针 
/// @param msg 消息字符数组，用于写入信息 
/// @param msg_len 指定返回消息的长度 默认1024
/// @return 返回分类标签
MY_DLL OCR_RES doInferenceBy3chImg(uchar* image_arr, const int height, const int width, const double conf, char* msg, size_t msg_len=1024);

/// @brief 根据 图像指针数组 进行标签分类
/// @param image_arr 图像内存指针 OpenCV BGR 3通道图的指针  TODO  图片数组！！！
/// @param height 图像高度  要求图片都相同尺寸
/// @param width 图像宽度  要求图片都相同尺寸
/// @param compiled_model OpenVINO 模型指针 
/// @param msg 消息字符数组，用于写入信息 
/// @param msg_len 指定返回消息的长度 默认1024
/// @return 返回分类标签
CLS_RES doInferenceBy3chBatchImgs(uchar** image_arr, const std::size_t img_num, const int height, const int width, void* compiled_model, char* msg, size_t msg_len=1024);


// TODO: 待实现
int doInferenceBatchImgs(const char* image_dir, int height, int width, void* compiled_model, const int* roi, const int roi_len, char* msg, size_t msg_len=1024);

void warmUp(void* compiled_model, char* msg, size_t msg_len=1024);
CLS_RES doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, char* msg, size_t msg_len=1024);
//char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img);
//char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor);
string getTimeNow();
