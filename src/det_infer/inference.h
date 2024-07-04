#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>

#define MY_DLL extern "C" __declspec(dllexport)

using std::string;
using std::cin;
using std::cout;
using std::endl;
using std::vector;


struct DET_RES{
    int tl_x;
    int tl_y;
    int br_x;
    int br_y;
    int cls = -1;
    double confidence;
    DET_RES()=default;
    DET_RES(const cv::Rect2d rec_, int cls_id_, double conf_):tl_x(static_cast<int>(rec_.x)), 
                                                            tl_y(static_cast<int>(rec_.y)),                                                                                             br_x(static_cast<int>(rec_.x+rec_.width)), 
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

MY_DLL void* initModel(const char* onnx_pth, char* msg);
MY_DLL void warmUp(void* model_ptr, char* msg);
MY_DLL DET_RES* doInferenceBy3chImg(uchar* image_arr, int height, int width, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg);
// 传入ROI 切分到合适的大小进行推理
MY_DLL DET_RES* doInferenceByImgPth(const char* img_pth, const int* roi, const int roi_len, void* model_ptr, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg);


DET_RES* doInferenceByImgMat(const cv::Mat& img_mat, void* compiled_model, const float score_threshold, const bool is_use_nms, size_t& det_num, char* msg);
char* resizeImageAsYOLO(ov::CompiledModel& compiled_model, const cv::Mat& org_img, cv::Mat& boarded_img, double& scale_ratio, int& left_padding, int& top_padding);
char* opencvMat2Tensor(cv::Mat& img_mat, ov::CompiledModel& compiled_model, ov::Tensor& out_tensor);
