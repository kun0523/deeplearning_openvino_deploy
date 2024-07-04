#include <iostream>

#include "inference.h"

void main(){

    cout << "opencv version: " << CV_VERSION << endl;

    string onnx_pth = R"(D:\share_dir\pd_edge_crack\workdir\runs\classify\train_yolos_freeze_use_aug_sgd2\weights\best.onnx)";
    char msg[1024];
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl;

    std::string image_dir{R"(../test_images)"};
    for(const auto& img_pth : std::filesystem::directory_iterator(image_dir)){
        cout << img_pth.path().string() << endl;
        cv::Mat img = cv::imread(img_pth.path().string());
        CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
        cout << msg << endl;
        cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
    }

    // getchar();
    
}