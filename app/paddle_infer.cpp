#include <iostream>
#include <filesystem>
#include<thread>
#include "inference.h"

/*
void testOcr(){
    std::cout << "Test Paddle OCR" << std::endl;

    char msg[1024]; 
    printInfo(msg, 1024);
    std::cout << msg << std::endl;
    
    std::string det_model_dir = R"(E:\Pretrained_models\OCR\det\en\en_PP-OCRv3_det_infer)";
    std::string cls_model_dir = R"(E:\Pretrained_models\OCR\cls\ch_ppocr_mobile_v2.0_cls_infer)";
    std::string rec_model_dir = R"(E:\Pretrained_models\OCR\rec\en\en_PP-OCRv4_rec_infer)";
    std::string rec_label_file = R"(D:\envs\paddleocr\Lib\site-packages\paddleocr\ppocr\utils\en_dict.txt)";
    std::string test_image = R"(E:\DataSets\pd_mix\gen_pcb\0_20240619_00001_B12_QRCODE=A742536058_12_17_C4_FK_A2FK4SD5OLPBA042_A2FK4S45KCFBA065.jpg)";

    initModel(det_model_dir.c_str(), cls_model_dir.c_str(), rec_model_dir.c_str(), rec_label_file.c_str(), 2, msg);
    OCR_RES result2 = doInferenceByImgPth(test_image.c_str(), 0.5, msg);
    // destroyModel();

    // // 会有内存泄漏问题，需要删除模型后回收内存资源
    // std::string img_dir = R"(E:\DataSets\pd_mix\20250308_new)";
    // for(const auto& file:std::filesystem::directory_iterator(img_dir)){
    //     auto subfix = file.path().extension().string();
    //     if(subfix==".jpg"||subfix==".png"||subfix==".bmp"){
    //         cout << file << endl;
    //         auto start = std::chrono::high_resolution_clock::now();
    //         // doInferenceByImgPth(file.path().string().c_str(), 0.5, msg);
    //         cv::Mat tmp = cv::imread(file.path().string());
    //         cv::resize(tmp, tmp, cv::Size(), 0.3, 0.3);
    //         doInferenceBy3chImg(tmp.data, tmp.rows, tmp.cols, 0.5, msg);
    //         auto end = std::chrono::high_resolution_clock::now();
    //         std::chrono::duration<double, std::milli> spend = end - start;
    //         cout << "infer cost: " << spend.count() << endl;
    //         // std::this_thread::sleep_for(std::chrono::milliseconds(2000));            
    //     }
    // }

    cv::Mat img = cv::imread(test_image);
    cv::resize(img, img, cv::Size(), 0.5, 0.5);  // 大概十几个像素高的识别效果还行
    cout << "image height: " << img.rows << " width: " << img.cols << endl;
    auto res2 = doInferenceBy3chImg(img.data, img.rows, img.cols, 0.5, msg);
    for(int i{}; i<res2.items.size(); ++i){
        auto bbox = res2.items[i].bbox;
        std::vector<cv::Point> points;
        for(int j{}; j<4; ++j){
            points.emplace_back(bbox[2*j], bbox[2*j+1]);
        }
        cv::polylines(img, points, true, cv::Scalar(255,255,255));
        cv::putText(img, std::to_string(i), cv::Point(bbox[0], bbox[1]), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,0,0));
        cout << i << " " << result2.items[i].words << "  H: " << bbox[5] - bbox[1] << endl;
    }
    // cv::imwrite("test.jpg", img);
    cv::imshow("test", img);
    cv::waitKey(0);

    destroyModel();
    // 已经删除模型后会报错
    OCR_RES res3 = doInferenceByImgPth(test_image.c_str(), 0.5, msg);
}
*/


void testDet(){
    cout << "Test Paddle OCR" << endl;

    char msg[1024]{};
    printInfo(msg, 1024);
    cout << msg << endl;

    std::string onnx_model_path = R"(D:\share_dir\pd_mix\workdir\det_pcb_320_img_no_mosaic\train\weights\best.onnx)";
    void* model_ptr = initModel(onnx_model_path.c_str(), msg, 1024);
    cout << msg << endl;
    cout << "model pointer: " << model_ptr << endl;

    

}

int main(){
    std::cout << "Paddle Test Demo" << std::endl;
    // testOcr();

    testDet();
    




    return 0;
}