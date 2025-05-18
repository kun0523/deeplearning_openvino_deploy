#ifndef ORT_BASE_H
#define ORT_BASE_H

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
    char* my_msg = nullptr;
    std::unique_ptr<Ort::Session> model_ptr = nullptr;
    // Ort::Session* model_ptr = nullptr;

    cv::Mat infer_img;
    std::stringstream msg_ss;
    void* result_ptr = nullptr;  // 推理结果数组指针
    int result_len{};  // 推理结果的个数
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
};


std::string getTimeNow();
void checkImageChannel(cv::Mat& input_mat);
cv::Scalar getRandomColor();

#endif