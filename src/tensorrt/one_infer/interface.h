#ifndef INTERFACE_H
#define INTERFACE_H

#include "base.h"

#define MY_DLL extern "C" __declspec(dllexport)

MY_DLL void* __stdcall initClsInfer(const char* model_pth_, char* msg);
MY_DLL void* __stdcall initDetInfer(const char* model_pth_, char* msg);
MY_DLL void* __stdcall initSegInfer(const char* model_pth_, char* msg);
MY_DLL void  __stdcall destroyInfer(Base* infer_);

MY_DLL void* __stdcall doInferenceByImgPath(Base* infer_, const char* img_pth, const int* roi, const float conf_threshold, int& det_num, char* msg);
MY_DLL void* __stdcall doInferenceByCharArray(Base* infer_, uchar* pixel_array, const int height, const int width, const float conf_threshold, int& det_num, char* msg);
MY_DLL void  __stdcall drawResult(Base* infer_, const short stop_period=0, const bool is_save=false);

#endif