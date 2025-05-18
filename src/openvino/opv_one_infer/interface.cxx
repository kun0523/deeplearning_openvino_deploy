#include "interface.h"

void* initClsInfer(const char* model_pth_, char* msg){
    Base* infer = new Classify(model_pth_, msg);
    return infer;
}

void* initDetInfer(const char* model_pth_, char* msg){
    Base* infer = new Detection(model_pth_, msg);
    return infer;
}

void* initSegInfer(const char* model_pth_, char* msg){
    Base* infer = new Segmentation(model_pth_, msg);
    return infer;
}

void destroyInfer(Base* infer_){
    delete infer_;
}

void* doInferenceByImgPath(Base* infer_, const char* img_pth, const int* roi, const float conf_threshold, int& det_num, char* msg){
    return infer_->inferByImagePath(img_pth, roi, conf_threshold, det_num, msg);

}

void* doInferenceByCharArray(Base* infer_, uchar* pixel_array, const int height, const int width, const float conf_threshold, int& det_num, char* msg){
    return infer_->inferByCharArray(pixel_array, height, width, conf_threshold, det_num, msg);
}

void drawResult(Base* infer_, const short stop_period, const bool is_save){
    infer_->drawResult(stop_period, is_save);
}


