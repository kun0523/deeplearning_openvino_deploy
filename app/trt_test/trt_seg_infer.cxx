#include "inference.h"

void testInferenceSpeed(){
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-fp16.engine)"};
    std::vector<std::string> image_files{
        // R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",  // 1560*1600  127ms
        // R"(E:\DataSets\imageNet\n02086240_Shih-Tzu.JPEG)",  // 1280*1024  49ms
        // R"(E:\le_trt\models\dog.jpg)",  // 500*374  26ms

        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)",
        R"(E:\le_trt\models\dog.jpg)"
    };
    std::vector<double> costs{};
    double total_costs{};
    char msg[1024];
    initModel(model_file.c_str(), msg);
    int idx{};
    for(const auto& file:image_files){
        int det_num{};
        auto start = std::chrono::high_resolution_clock::now();
        auto r = doInferenceByImgPth(file.c_str(), nullptr, 0.5f, det_num, msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> spend = end - start;
        total_costs += spend.count();
        costs.push_back(spend.count());
        std::cout << "Result: " << r->get_info() << std::endl;
        std::cout << idx++ << " cost: " << spend.count() << "ms" << std::endl;
        // destroyModel();
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
}


int main(){
    std::cout << "TensorRT Test Demo" << std::endl;

    testInferenceSpeed(); 
    return 0;

    printInfo();

    // run();

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-fp16.engine)"};
    // std::string img_file{R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)"};
    std::string img_file{R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"};

    // initModel
    char msg[1024];
    initModel(onnx_file.c_str(), msg);
    std::cout << msg << std::endl;

    // doInferenceByMat
    cv::Mat img = cv::imread(img_file);
    int det_num{};
    SEG_RES* res = doInferenceByImgMat(img, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res[i].get_info() << std::endl;
    
    return 0;

    // doInferenceByImgPth()
    SEG_RES* res2 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.3f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res2[i].get_info() << std::endl;

    // doInferenceBy3chImg()
    SEG_RES* res3 = doInferenceBy3chImg(img.data, img.rows, img.cols, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res3[i].get_info() << std::endl;

    // speed test...
    testInferenceSpeed();
}