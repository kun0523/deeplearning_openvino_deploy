#include <iostream>
#include "inference.h"

// #define CLS
#define DET 
// #define SEG 

#ifdef CLS 
void testClsInfer(){
    cout << "----- Test Classification API -----" << endl;

    // 接口 1：模型初始化
    string onnx_pth = R"(D:\share_dir\pd_edge_crack\workdir\runs\classify\train_yolos_freeze_use_aug_sgd2\weights\best.onnx)";
    char msg[1024];
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl;

    // 接口 2：指定图片路径推理
    std::string img_pth = R"(E:\my_depoly\bin\test_images\cls_crack_test2.jpg)";
    auto result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, nullptr, msg);
    cout << "Got Class: " << result2.cls << " Confidence: " << result2.confidence << endl;
    cout << msg << endl;

    // 接口 3：传图片指针推理
    cv::Mat img = cv::imread(img_pth);
    CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    cout << msg << endl;
    cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
    // std::string image_dir{R"(../test_images)"};
    // for(const auto& img_pth : std::filesystem::directory_iterator(image_dir)){
    //     cout << img_pth.path().string() << endl;
    //     cv::Mat img = cv::imread(img_pth.path().string());
    //     CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    //     cout << msg << endl;
    //     cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
    // }
}
#endif

#ifdef DET 
void testDetInfer(){
    cout << "----- Test Detection API -----" << endl;
    // 接口 1：模型初始化
    // string onnx_pth = R"(D:\share_dir\cell_det\workdir\runs\detect\det_s_freeze10_sgd\weights\best.onnx)";  // det cell yolov8
    // bool use_nms = true;
    // std::string img_pth = R"(E:\my_depoly\bin\test_images\det_cell_test3.jpg)";

    string onnx_pth = R"(D:\share_dir\impression_detect\workdir\yolov10\dent_det\yolov10s_freeze8_use_sgd2\weights\yolov10_freeze8_best.onnx)";
    // string onnx_pth = R"(D:\share_dir\impression_detect\workdir\yolov10\dent_det\yolov10s_freeze8_use_sgd2\weights\best.onnx)";
    bool use_nms = false;
    std::string img_pth = R"(E:\my_depoly\bin\test_images\det_cell_test4.jpg)";

    char msg[10240];
    std::memset(msg, '\0', 10240);
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl;

    // // 接口 2：指定图片路径推理
    // size_t det_num;
    // // TODO 推理的结果不对！！！！
    // DET_RES* result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, nullptr, 0.5f, use_nms, det_num, msg);
    // cout << msg << endl;
    // cout << "Got Detection Res: " << result2[0].get_info() << endl;

    // // 接口 3：传图片指针推理
    // cv::Mat img = cv::imread(img_pth);
    // DET_RES* res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, 0.5f, use_nms, det_num, msg);
    // cout << msg << endl;
    // for(int i=0; i<det_num; ++i){
    //     cout << "i: " << i << endl;
    //     cout << res[i].get_info() << endl;
    // }

    // 接口 4：图片分块多线程推理  单线程 1700+ms
    auto tick = std::chrono::high_resolution_clock::now();
    cv::Mat img_cell = cv::imread(img_pth);
    size_t det_num{0};
    int patch_size = 1000;
    int overlap_size = 100;
    DET_RES* res = doInferenceBy3chImgPatches(img_cell.data, img_cell.rows, img_cell.cols, patch_size, overlap_size, model_ptr, 0.5f, use_nms, det_num, msg);
    auto tock = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend = tock-tick;
    cout << "cost time: " << spend.count() << "ms" << endl;
    for(int i =0; i<det_num; ++i){
        cout << res[i].get_info() << endl;
    }
    // cout << msg << endl;
}
#endif 

#ifdef SEG 
void testSegInfer(){}
#endif


// TODO 留一个参数 控制是否输出Log文件
void main(){

    cout << "opencv version: " << CV_VERSION << endl;

    #ifdef CLS 
    testClsInfer();
    #endif 
    
    #ifdef DET 
    testDetInfer();
    #endif 

    #ifdef SEG 
    testSegInfer();
    #endif 


    // getchar();
    
}