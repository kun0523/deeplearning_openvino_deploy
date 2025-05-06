#ifndef ROOT_H
#define ROOT_H

#include <iostream>
#include <vector>
#include <functional>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <NvInfer.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <iomanip>
#include <ctime>
#include <iterator>

using namespace nvinfer1;


struct CLS_RES{
    short cls{-1};
    double confidence{-1};

    CLS_RES(short cls_, double confidence_):cls(cls_), confidence(confidence_) {}
    std::string get_info();
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

    // TODO: 写析构函数 验证是否有触发
    ~SEG_RES();
};

class Base{
public:
    Base(const char* model_pth_, char* msg);
    virtual void* inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& num, char* msg)=0;
    virtual void* inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& num, char* msg)=0;
    virtual ~Base();
    virtual void drawResult(const short stop_period=0, const bool is_save=false) const=0;

protected:
    const char* model_pth = nullptr;
    IRuntime* runtime = nullptr;
    ICudaEngine* engine = nullptr;
    IExecutionContext* context = nullptr;

    char* my_msg = nullptr;
    cv::Mat infer_img;
    std::stringstream msg_ss;
    void* result_ptr = nullptr;
    int result_len{};
};

class Classify: public Base{
public:
    Classify(const char* model_pth_, char* msg);
    void* inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& num, char* msg) override;
    void* inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& num, char* msg) override;
    ~Classify() override;
    void drawResult(const short stop_period=0, const bool is_save=false) const override;

private:
    void* inferByMat(cv::Mat& img_mat, const float conf_threshold, int& num, char* msg);

    void* io_buffer[2] = {NULL, NULL};
    float* outputHostBuffer = nullptr;
};

class Detection: public Base{
public:
    Detection(const char* model_pth_, char* msg);
    void* inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg) override;
    void* inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg) override;
    ~Detection() override;
    void drawResult(const short stop_period=0, const bool is_save=false)const override;

private:
    void* inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg);
    void warmUp(char* msg);
    void preProcess(const cv::Mat& org_img, cv::Mat& boarded_img);
    DET_RES* postProcess(const float conf_threshold, cv::Mat& det_result_mat, const double& scale_ratio_, int& det_num);

    void* io_buffer[2] = {NULL, NULL};
    float* outputHostBuffer = nullptr;
};

class Segmentation: public Base{
public:
    Segmentation(const char* model_pth_, char* msg);
    void* inferByImagePath(const char* image_pth, const int* roi, const float conf_threshold, int& det_num, char* msg) override;
    void* inferByCharArray(uchar* image_arr, const int height, const int width, const float conf_threshold, int& det_num, char* msg) override;
    ~Segmentation() override;
    void drawResult(const short stop_period=0, const bool is_save=false)const override;

private:
    void* inferByMat(cv::Mat& img_mat, const float conf_threshold, int& det_num, char* msg);
    void warmUp(char* msg);
    void preProcess(const cv::Mat& org_img, cv::Mat& boarded_img);
    SEG_RES* postProcess(const float conf_threshold, const cv::Mat& pred_mat, const cv::Mat& proto_mat, const cv::Size& org_size, const cv::Size& infer_size, int& det_num);

    void* io_buffer[3] = {NULL, NULL, NULL};
    float* outputHostBuffer_1 = nullptr;
    float* outputHostBuffer_2 = nullptr;

};


std::string getTimeNow();
void checkImageChannel(cv::Mat& input_mat);

#endif