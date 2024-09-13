#include <iostream>
#include "inference.h"
#include <filesystem>

#define CLS
// #define DET 
// #define SEG 

void createDirectoryIfNotExists(const std::string& dirPath) {
    // 检查目录是否存在
    if (!std::filesystem::exists(dirPath)) {
        // 尝试创建目录
        try {
            std::filesystem::create_directories(dirPath);
            std::cout << "Success to create save dir: " << dirPath << std::endl;
        } catch (const std::filesystem::filesystem_error& e) {
            std::cerr << "Failed to create save dir: " << e.what() << std::endl;
        }
    } else {
        std::cout << "Save Dir Already Exists: " << dirPath << std::endl;
    }
}

#ifdef CLS 
void testClsInfer(){
    cout << "----- Test Classification API -----" << endl;

    // 接口 1：模型初始化
    // string onnx_pth = R"(D:\share_dir\pd_edge_crack\workdir\runs\classify\train_yolos_freeze8_sgd_aug2\weights\classify_crack_model_0721.onnx)";
    string onnx_pth;
    cout << "Input Model Onnx Path: ";
    cin >> onnx_pth;
    cout << endl;
    char msg[1024];
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl; 

    // 创建保存路径
    string save_dir;
    cout << "Save Result Dir: ";
    cin >> save_dir;
    cout << endl;
    string save_dir_ng = save_dir + "/ng";
    string save_dir_ok = save_dir + "/ok";
    createDirectoryIfNotExists(save_dir_ng);
    createDirectoryIfNotExists(save_dir_ok);

    // 选择推理模式： 单张推理 or 遍历文件夹
    string infer_mode;
    cout << "Select Inference Mode: 1: one pic to infer; 2: loop directory to infer: ";
    cin >> infer_mode;
    if(infer_mode == "1"){
        // std::string img_pth = R"(E:\DataSets\edge_crack\tmp\test\_489_725_202406291324116_0.jpg)";
        // 接口 2：指定图片路径推理
        std::string img_pth;
        while(true){
            cout << "Input Image Path: ";
            cin >> img_pth;   
            
            int roi[] = {350, 10, 1550, 810};
            auto result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, roi, msg);
            cout << "Got Class: " << result2.cls << " Confidence: " << result2.confidence << endl;
            cout << msg << endl;
            std::filesystem::path tmp_pth(img_pth);
            string tmp_save_pth;
            if(result2.cls==0)
                tmp_save_pth = save_dir_ng + "/" + tmp_pth.filename().string();
            else
                tmp_save_pth = save_dir_ok + "/" + tmp_pth.filename().string();

            std::filesystem::copy_file(img_pth, tmp_save_pth, std::filesystem::copy_options::overwrite_existing);
        }
    }else if(infer_mode == "2"){
        // 接口 2：遍历图片文件夹推理 
        // std::string img_dir = R"(E:\DataSets\edge_crack\tmp\test\)";
        // std::string save_dir = R"(E:\DataSets\edge_crack\tmp\crack\)";
        string img_dir;
        cout << "Input Source Image dir:";
        cin >> img_dir;
        cout << endl;
        int total_counter = 0, ng_counter = 0;
        for(auto& img_pth : std::filesystem::directory_iterator(img_dir)){
            cout << "Now Process Image: " << img_pth.path().filename() << endl;
            cv::Mat img = cv::imread(img_pth.path().string());
            cv::Mat img_enhance;
            img.convertTo(img_enhance, CV_8UC3, 1.0, 0);
            total_counter++;

            std::filesystem::path tmp_pth(img_pth);
            string tmp_save_pth;

            auto result = doInferenceBy3chImg(img_enhance.ptr(), img.rows, img.cols, model_ptr, msg);
            cout << "Got Class: " << result.cls << " Confidence: " << result.confidence << endl;
            if (result.cls == 0){
                // cv::imwrite(save_dir+"/"+img_pth.path().filename().string(), img);
                tmp_save_pth = save_dir_ng + "/" + tmp_pth.filename().string();
                ng_counter++;
            }
            else{
                tmp_save_pth = save_dir_ok + "/" + tmp_pth.filename().string();
            }
            std::filesystem::copy_file(img_pth, tmp_save_pth, std::filesystem::copy_options::overwrite_existing);

        }
        cout << "Totall Num: " << total_counter << " NG Num: " << ng_counter << endl;
        
    }
    else{
        cout<< "Wrong Input: " << infer_mode << " Only Support: 1: one pic to infer; 2: loop directory to infer;\n";
    }

    // // 接口 3：传图片指针推理
    // cv::Mat img = cv::imread(img_pth);
    // CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    // cout << msg << endl;
    // cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
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
    // 产品区间检测
    // string onnx_pth = R"(D:\share_dir\cell_det\workdir\runs\detect\det_s_freeze10_sgd\weights\best.onnx)";  // det cell yolov8
    // bool use_nms = true;
    // std::string img_pth = R"(E:\my_depoly\bin\test_images\det_cell_test3.jpg)";

    // 破碎检查
    // D:\share_dir\pd_edge_crack\workdir\det_crack_rect\yolom_freeze9_sgd_aug4\weights\yolov8_det02.onnx
    // E:\DataSets\edge_crack\cut_patches_0828\crack\2024082215300_D3_unk_3480.jpg
    string onnx_pth;
    cout << "Onnx Model Path: ";
    cin >> onnx_pth; // R"(D:\share_dir\pd_edge_crack\workdir\det_crack\yolom_freeze9_sgd_aug6\weights\yolov8_det02.onnx)";
    bool use_nms = true;
    std::string img_pth; // R"(E:\DataSets\edge_crack\cut_patches_0825\tmp\20240628092227_1371.jpg)";
    cout << "Image Path: ";
    cin >> img_pth;

    // testAsync();

    char msg[10240];
    std::memset(msg, '\0', 10240);
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl;

    // 接口 2：指定图片路径推理
    size_t det_num;
    auto start = std::chrono::high_resolution_clock::now();
    DET_RES* result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, nullptr, 0.3f, use_nms, det_num, msg);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend = end -start;
    cout << msg << endl;
    cout << "Got Detection Res: " << result2[0].get_info() << endl;
    cout << "Cost: " << spend.count() << "ms" << endl;

    // // 接口 3：传图片指针推理
    // cv::Mat img = cv::imread(img_pth);
    // // DET_RES* res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, 0.5f, use_nms, det_num, msg);
    // // cout << msg << endl;
    // // for(int i=0; i<det_num; ++i){
    // //     cout << "i: " << i << endl;
    // //     cout << res[i].get_info() << endl;
    // // }
    // CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    // cout << msg << endl;
    // cout << "class: " << res.cls << " confidence: " << res.confidence << endl;

    // // 接口 4：图片分块多线程推理  单线程 1700+ms
    // auto tick = std::chrono::high_resolution_clock::now();
    // cv::Mat img_cell = cv::imread(img_pth);
    // size_t det_num{0};
    // int patch_size = 1000;
    // int overlap_size = 100;
    // DET_RES* res = doInferenceBy3chImgPatches(img_cell.data, img_cell.rows, img_cell.cols, patch_size, overlap_size, model_ptr, 0.5f, use_nms, det_num, msg);
    // auto tock = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> spend = tock-tick;
    // cout << "cost time: " << spend.count() << "ms" << endl;
    // for(int i =0; i<det_num; ++i){
    //     cout << res[i].get_info() << endl;
    // }
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