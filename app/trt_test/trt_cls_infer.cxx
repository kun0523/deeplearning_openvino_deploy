#include "inference.h"

void testInferenceSpeed(){
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls-fp16.engine)"};
    std::vector<std::string> image_files{
        // R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",  // 1560*1600  168ms
        // R"(E:\DataSets\imageNet\n02086240_Shih-Tzu.JPEG)",  // 1280*1024  78ms
        // R"(E:\le_trt\models\dog.jpg)",  // 500*374  65ms

        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",
        R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"
    };
    std::vector<double> costs{};
    double total_costs{};
    char msg[1024];
    initModel(model_file.c_str(), msg);
    int idx{};
    for(const auto& file:image_files){
        int det_num{};
        auto start = std::chrono::high_resolution_clock::now();
        auto r = doInferenceByImgPth(file.c_str(), nullptr, msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> spend = end - start;
        total_costs += spend.count();
        costs.push_back(spend.count());
        std::cout << "Result: " << r.cls << " " << r.confidence << std::endl;
        std::cout << idx++ << " cost: " << spend.count() << "ms" << std::endl;
        // destroyModel();
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
}


int main(){
    std::cout << "TensorRT Classification Demo" << std::endl;

    testInferenceSpeed();
    return 0;

    printInfo();

    run();

    std::cout << "..................." << std::endl;
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11-cls\yolo11n-cls-fp16.engine)"};
    char msg[1024];
    initModel(model_file.c_str(), msg);
    std::string img_file{R"(E:\DataSets\imageNet\n01443537_goldfish.JPEG)"};
    cv::Mat img = cv::imread(img_file);
    CLS_RES r1 = doInferenceByImgMat(img, msg);
    std::cout << r1.cls << " " << r1.confidence << std::endl;
    std::cout << msg << std::endl;

    // destroyModel();  // 删除后还是可以推理。。。

    std::cout << "..................." << std::endl;
    CLS_RES r2 = doInferenceByImgPth(img_file.c_str(), nullptr, msg);
    std::cout << r2.cls << " " << r2.confidence << std::endl;

    std::cout << "..................." << std::endl;
    CLS_RES r3 = doInferenceBy3chImg(img.ptr(), img.rows, img.cols, msg);
    std::cout << r3.cls << " " << r3.confidence << std::endl;

}