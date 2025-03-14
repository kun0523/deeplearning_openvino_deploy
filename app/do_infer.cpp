#include <iostream>
#include "inference.h"
#include <filesystem>
#include <regex>

// #define PADDCLS
// #define PADDDET
// #define SKLEARN
// #define CLS
#define DET 
// #define DET_OPVNO
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

#ifdef PADDDET
void testPaddleClsInfer(){
    std::cout << "----- Test Paddle Det API ------" << std::endl;    
    char msg[1024]; 
    printInfo(msg, 1024);
    std::cout << msg;
    }
#endif

#ifdef PADDCLS
void testPaddleClsInfer(){
    std::cout << "----- Test Paddle Cls API ------" << std::endl;    
    char msg[1024]; 
    printInfo(msg, 1024);
    std::cout << msg;

    string model_dir = R"(E:\le_paddlex\crack_cls\output_hgnetv2_b1\best_model\inference)";    // HGNet
    // string model_dir = R"(E:\le_paddlex\crack_cls\output_lcnetv2_small\best_model\inference)";   // LCNet  
    // string model_dir = R"(E:\le_paddlex\crack_cls\output_ResNet50_vd\best_model\inference)";   // ResNet 
    auto model_ptr = initModel(model_dir.c_str(), 2, msg);
    printf("model_ptr: %p\n", model_ptr);


    string img_pth = R"(E:\DataSets\tuffy_crack\pdx_clas\images\20241027061702533.jpg)";
    doInferenceByImgPth(img_pth.c_str(), model_ptr, msg);

    auto start = std::chrono::high_resolution_clock::now();
    doInferenceByImgPth(img_pth.c_str(), model_ptr, msg);
    auto end = std::chrono::high_resolution_clock::now();
    auto spend = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    printf("Paddle Class Inference Cost: %d ms\n", spend.count());

    cv::Mat img = cv::imread(img_pth);
    doInferenceBy3chImg(img.ptr(), img.rows, img.cols, model_ptr, msg);

}
#endif

#ifdef DET_OPVNO
void testOpenvinoDetInfer(){
    cout << "----- Test OpenVINO Det API -----" << endl;

    initModel(R"(D:\share_dir\impression_detect\workdir\yolov11\det_dent_gold_scf\yolo11s_sgd3\weights\best_int8_openvino_model)");


    // doInferenceByImgPth();

}
#endif


#ifdef SKLEARN
void testSklearnInfer(){
    std::cout << "------ Test Sklearn infer API -------" << std::endl;

    std::string onnx_pth{};
    std::cout << "Input Onnx Model Path: ";
    std::cin >> onnx_pth;
    // std::string onnx_pth = R"(D:\share_dir\cell_corner_curve\src\random_forest_5feats.onnx)";
    void* session_p = initModel(onnx_pth.c_str());
    std::cout << "========== First Inference lb =============" << std::endl;
    Point points[5] = {Point(3.718,-1.332), Point(3.082,-1.789), Point(2.496,-2.308), Point(1.981,-2.897), Point(1.535,-3.536)};
    CLS_RES result = doInference(session_p, Point(9.803, -9.484), points, 5);
    result.print();

    std::cout << "========== Second Inference lt =============" << std::endl;
    Point points2[5] = {Point(1.306,3.438), Point(1.775,2.782), Point(2.29,2.157), Point(2.913,1.639), Point(0,0)};
    result = doInference(session_p, Point(9.553, 9.878), points2, 5);
    result.print();  

    std::cout << "========== Third Inference rb =============" << std::endl;
    Point points3[5] = {Point(-1.392, -3.307), Point(-1.891, -2.657), Point(-2.473, -2.079), Point(-3.069, -1.511), Point()};
    result = doInference(session_p, Point(-9.566, -9.796), points3, 5);
    result.print();   

    std::cout << "========== Forth Inference rt =============" << std::endl;
    Point points4[5] = {Point(-1.392, -3.307), Point(-1.891, -2.657), Point(-2.473, -2.079), Point(-3.069, -1.511), Point()};
    result = doInference(session_p, Point(-9.566, -9.796), points4, 5);
    result.print();   
}
#endif

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
    std::string img_pth;
    string infer_mode;
    cout << "Select Inference Mode: 1: one pic to infer; 2: loop directory to infer: ";
    cin >> infer_mode;
    if(infer_mode == "1"){
        // std::string img_pth = R"(E:\DataSets\edge_crack\tmp\test\_489_725_202406291324116_0.jpg)";
        // 接口 2：指定图片路径推理
        while(true){
            cout << "Input Image Path: ";
            cin >> img_pth;   
            
            // int roi[] = {350, 10, 1550, 810};
            auto result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, nullptr, msg);
            cout << "Got Class: " << result2.cls << " Confidence: " << result2.confidence << endl;
            cout << msg << endl;
            std::filesystem::path tmp_pth(img_pth);
            string tmp_save_pth;
            if(result2.cls==0 || result2.cls==2)
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
            cv::Mat img = cv::imread(img_pth.path().string(), cv::ImreadModes::IMREAD_COLOR);
            cv::Mat img_enhance;            
            cv::convertScaleAbs(img, img_enhance, 1.0, 0.0);
            // cv::convertScaleAbs(img, img_enhance, 4, 10);
            // cv::imshow("test", img_enhance);
            // cv::waitKey(0);
            total_counter++;

            std::filesystem::path tmp_pth(img_pth);
            string tmp_save_pth;

            auto result = doInferenceBy3chImg(img_enhance.ptr(), img.rows, img.cols, model_ptr, msg);
            cout << "Got Class: " << result.cls << " Confidence: " << result.confidence << endl;
            if (result.cls == 0 || result.cls==2){
                // cv::imwrite(save_dir+"/"+img_pth.path().filename().string(), img);
                tmp_save_pth = save_dir_ng + "/" + tmp_pth.filename().string();
                ng_counter++;
            }
            else{
                tmp_save_pth = save_dir_ok + "/" + tmp_pth.filename().string();
            }
            // std::filesystem::copy_file(img_pth, tmp_save_pth, std::filesystem::copy_options::overwrite_existing);
            cv::imwrite(tmp_save_pth, img_enhance);

        }
        cout << "Totall Num: " << total_counter << " NG Num: " << ng_counter << endl;
        
    }
    else{
        cout<< "Wrong Input: " << infer_mode << " Only Support: 1: one pic to infer; 2: loop directory to infer;\n";
    }

    // 接口 3：传图片指针推理
    cv::Mat img = cv::imread(img_pth);
    CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    cout << msg << endl;
    cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
    std::string image_dir{R"(../test_images)"};
    for(const auto& img_pth : std::filesystem::directory_iterator(image_dir)){
        cout << img_pth.path().string() << endl;
        cv::Mat img = cv::imread(img_pth.path().string());
        CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
        cout << msg << endl;
        cout << "Class: " << res.cls << " Confidence: " << res.confidence << endl;
    }
}

void imageEnhance(){

    string org_img_dir{}, save_img_dir{};
    while(true){

        cout << "Input Image Directory: ";
        cin >> org_img_dir;
        cout << "Input Save Directory: ";
        cin >> save_img_dir;

        auto d = std::filesystem::path(org_img_dir);    
        string cls = d.filename().string();
        if(!std::filesystem::exists(org_img_dir)){
            cout << org_img_dir << " not Exists!" << endl;
            continue;
        }

        if(!std::filesystem::exists(save_img_dir)){
            if(!std::filesystem::create_directories(std::filesystem::path(save_img_dir))){
                cout << save_img_dir << " Create Failed!"<<endl;
                continue;
            }
        }

        for(auto file : std::filesystem::directory_iterator(org_img_dir)){            
            cout << "file: " << file << endl;
            cv::Mat org_img = cv::imread(file.path().string(), cv::IMREAD_COLOR);

            cv::Mat res_img;
            cv::convertScaleAbs(org_img, res_img, 4, 10);

            // cv::imshow(cls, res_img);
            // cv::waitKey(0);

            string save_pth = save_img_dir + "\\" + file.path().filename().string();
            cv::imwrite(save_pth, res_img);
        }
    }

}
#endif

#ifdef DET 
void testDetInfer(){
    cout << "----- Test Detection API -----" << endl;
    // 接口 1：模型初始化
    string onnx_pth{};
    cout << "Onnx Model Path: ";
    cin >> onnx_pth; // R"(D:\share_dir\pd_edge_crack\workdir\det_crack\yolom_freeze9_sgd_aug6\weights\yolov8_det02.onnx)";
    std::string model_type_s{};
    cout << "Model Type: ";
    cin >> model_type_s;
    short model_type = std::stoi(model_type_s);
    std::string img_pth{}; // R"(E:\DataSets\edge_crack\cut_patches_0825\tmp\20240628092227_1371.jpg)";
    cout << "Image Path: ";
    cin >> img_pth;

    char msg[10240];
    std::memset(msg, '\0', 10240);
    void* model_ptr = initModel(onnx_pth.data(), msg);
    cout << msg << endl;

    // 接口 2：指定图片路径推理
    cout << ">>>>>>>>>>>>> Inference by image path <<<<<<<<<<<<<<<<<<<" << endl; 
    for(int i{}; i<10; i++){
    size_t det_num;
    auto start = std::chrono::high_resolution_clock::now();
    DET_RES* result2 = doInferenceByImgPth(img_pth.c_str(), model_ptr, nullptr, 0.3f, model_type, det_num, msg);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> spend = end -start;
    cout << msg << endl;

    // cv::Mat org_img = cv::imread(img_pth);
    // for(size_t i{0}; i<det_num; ++i){
    //     cout << "Got Detection Res: " << result2[i].get_info() << endl;        
    //     cv::rectangle(org_img, cv::Rect2d(cv::Point2d(result2[i].tl_x, result2[i].tl_y), cv::Point2d(result2[i].br_x, result2[i].br_y)), cv::Scalar(0, 255, 0), 20);
    // }
    cout << "Cost: " << spend.count() << "ms" << endl;
    }
    // cv::resize(org_img, org_img, cv::Size(), 0.2, 0.2);
    // cv::imshow("Test", org_img);
    // cv::waitKey(0);

    // // 接口 3：传图片指针推理
    // cout << ">>>>>>>>>>>>> Inference by image pointer <<<<<<<<<<<<<<<<<<<" << endl; 
    // cv::Mat img = cv::imread(img_pth);
    // DET_RES* res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, 0.5f, model_type, det_num, msg);
    // cout << msg << endl;
    // for(int i=0; i<det_num; ++i){
    //     cout << "i: " << i << endl;
    //     cout << res[i].get_info() << endl;
    // }
    // CLS_RES res = doInferenceBy3chImg(img.data, img.rows, img.cols, model_ptr, msg);
    // cout << msg << endl;
    // cout << "class: " << res.cls << " confidence: " << res.confidence << endl;

    // 测试 v8 v10 v11 版本模型效果
    // char msg[10240];
    // size_t det_num{};
    // cout << "================= yolov8 ====================" << endl;
    // std::memset(msg, '\0', 10240);
    // std::string v8_onnx = R"(D:\share_dir\cell_det\workdir\runs\detect\det_s_freeze10_sgd\weights\det_cell_s_0805.onnx)";
    // std::string v8_img_pth = R"(E:\DataSets\dents_det\org_D1\gold_scf\NG\20231205_00001_P51-R_1_16_C2_DS_A2DS2S39593AC028_A2DS2S3903IBE011.jpg)";
    // void* model_ptrv8 = initModel(v8_onnx.data(), msg);
    // cout << msg << endl;
    // DET_RES* resultv8 = doInferenceByImgPth(v8_img_pth.c_str(), model_ptrv8, nullptr, 0.3f, 8, det_num, msg);
    // cout << msg << endl;
    // for(int i=0; i<det_num; ++i){
    //     cout << "i: " << i << " ";
    //     cout << resultv8[i].get_info() << endl;
    // }

    // cout << "================= yolov10 ====================" << endl;    
    // std::memset(msg, '\0', 10240);
    // std::string v10_onnx = R"(D:\share_dir\impression_detect\workdir\yolov10\yolov10m\d1_black_sgd\weights\yolov10m_01.onnx)";
    // std::string v10_img_pth = R"(E:\DataSets\dents_det\org_D1\black_scf\cutPatches\NG\20240925_00001_P51-L_A743934612_1_17_C2_FS_A2FS1S46YR9DC140_A2FS1R4662RAD044_8391.jpg)";
    // void* model_ptrv10 = initModel(v10_onnx.data(), msg);
    // cout << msg << endl;
    // DET_RES* resultv10 = doInferenceByImgPth(v10_img_pth.c_str(), model_ptrv10, nullptr, 0.3f, 10, det_num, msg);
    // cout << msg << endl;
    // for(int i=0; i<det_num; ++i){
    //     cout << "i: " << i << " ";
    //     cout << resultv10[i].get_info() << endl;
    // }

    // cout << "================= yolov11 ====================" << endl;
    // std::memset(msg, '\0', 10240);
    // std::string v11_onnx = R"(D:\share_dir\impression_detect\workdir\yolov11\det_dent_gold_scf\yolov11m_sgd\weights\yolov11m_04.onnx)";
    // std::string v11_img_pth = R"(E:\DataSets\dents_det\org_D1\gold_scf\cutPatches\NG\5_4183.jpg)";
    // void* model_ptrv11 = initModel(v11_onnx.data(), msg);
    // cout << msg << endl;
    // DET_RES* resultv11 = doInferenceByImgPth(v11_img_pth.c_str(), model_ptrv11, nullptr, 0.3f, 11, det_num, msg);
    // cout << msg << endl;
    // for(int i=0; i<det_num; ++i){
    //     cout << "i: " << i << " ";
    //     cout << resultv11[i].get_info() << endl;
    // }


    // 接口 4：图片分块多线程推理  单线程 1700+ms
    // string onnx_pth{};
    // cout << "Onnx Model Path: ";
    // cin >> onnx_pth; // R"(D:\share_dir\pd_edge_crack\workdir\det_crack\yolom_freeze9_sgd_aug6\weights\yolov8_det02.onnx)";
    // std::string model_type_s{};
    // cout << "Model Type: ";
    // cin >> model_type_s;
    // short model_type = std::stoi(model_type_s);
    // std::string img_dir{}; // R"(E:\DataSets\edge_crack\cut_patches_0825\tmp\20240628092227_1371.jpg)";
    // cout << "Image Dir: ";
    // cin >> img_dir;

    // 同步推理与异步推理对比   20pcs 异步快 300+ms  异步推理有bug！！！
    // char msg[10240];
    // std::memset(msg, '\0', 10240);
    // std::string v11_onnx = R"(D:\share_dir\impression_detect\workdir\yolov11\det_dent_gold_scf\yolo11s_sgd3\weights\best.onnx)";
    // std::string img_dir = R"(E:\DataSets\dents_det\org_D1\gold_scf\cutPatches640\NG)";
    // void* model_ptrv11 = initModel(v11_onnx.data(), msg);
    // size_t det_num{};
    // size_t img_num{20};
    // size_t img_counter{};

    // std::vector<DET_RES*> sync_res{};
    // auto sync_tick = std::chrono::high_resolution_clock::now();
    // for(auto f:std::filesystem::directory_iterator(img_dir)){
    //     if(f.path().extension() != ".jpg") 
    //         continue;

    //     std::cout << "img_name: " << f.path().filename() << std::endl;
    //     sync_res.push_back(doInferenceByImgPth(f.path().string().c_str(), model_ptrv11, nullptr, 0.3f, 11, det_num, msg));
    //     if(++img_counter>=img_num) break;
    // }
    // auto sync_tock = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> sync_spend = sync_tock - sync_tick;
    // cout << "cost time of sync inference: " << sync_spend.count() << "ms" << endl;
    // for(int i{}; i<sync_res.size(); ++i){
    //     std::cout << i << " " << sync_res[i]->get_info() << std::endl;
    // }

    // std::vector<uchar*> img_vec;
    // size_t img_rows{}, img_cols{};
    // img_counter = 0;
    // for( auto f : std::filesystem::directory_iterator(img_dir)){
    //     if(f.path().extension() != ".jpg") 
    //         continue;
    //     std::cout << "img_name: " << f.path().filename() << std::endl;
    //     cv::Mat* im = new cv::Mat(cv::imread(f.path().string(), cv::ImreadModes::IMREAD_COLOR));
    //     img_rows = im->rows;
    //     img_cols = im->cols;
    //     img_vec.push_back(im->ptr());

    //     if(++img_counter>=img_num)
    //         break;
    // }     

    // auto async_tick = std::chrono::high_resolution_clock::now();
    // void* res = doInferenceBy3chImgPatches(img_vec.data(), img_num, img_rows, img_cols, model_ptrv11, 0.1f, 11, msg);
    // auto async_tock = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double, std::milli> async_spend = async_tock - async_tick;
    // cout << "cost time of async inference: " << async_spend.count() << "ms" << endl;

    // auto vec = static_cast<std::vector<std::vector<DET_RES>>* >(res);

    // for( int i{}; i<vec->size(); ++i){        
    //     for (auto e:(*vec)[i]){
    //         std::cout<< i << " " << e.get_info() << std::endl;
    //     }

    // }

    // for(size_t i{}; i<img_vec.size(); ++i){
    //     delete img_vec[i];
    // }
    // cout << msg << endl;
}
#endif 

#ifdef SEG 
void testSegInfer(){}
#endif


// TODO 留一个参数 控制是否输出Log文件
void main(){

    // std::cout << "opencv version: " << CV_VERSION << std::endl;

    std::cout << "Start App: "<< std::endl;

    #ifdef PADDCLS
    testPaddleClsInfer();
    #endif

    #ifdef DET_OPVNO
    testOpenvinoDetInfer();
    #endif

    #ifdef SKLEARN
    testSklearnInfer();
    #endif

    #ifdef CLS 
    testClsInfer();

    // imageEnhance();
    #endif 
    
    #ifdef DET 
    testDetInfer();
    #endif 

    #ifdef SEG 
    testSegInfer();
    #endif 


    // getchar();
    
}