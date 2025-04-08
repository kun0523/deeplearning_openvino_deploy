#include "inference.h"

void testInferenceSpeed(){
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp16.engine)"};
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

    testInferenceSpeed();  // 30ms
    return 0;

    printInfo();

    // run();

    std::cout << "..................." << std::endl;
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp16.engine)"};
    char msg[1024];
    initModel(model_file.c_str(), msg);

    std::string img_file{R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"};
    cv::Mat img = cv::imread(img_file);
    int det_num{};
    std::cout << "-------------------" << std::endl;
    DET_RES* r1 = doInferenceByImgMat(img, 0.5f, det_num, msg);
    std::cout << msg << std::endl;
    std::cout << "det_num: " << det_num << std::endl;
    for(int i{}; i<det_num; ++i){
        std::cout << r1[i].get_info() << std::endl;
        cv::Rect r(cv::Point(r1[i].tl_x, r1[i].tl_y), cv::Point(r1[i].br_x, r1[i].br_y));
        cv::rectangle(img, r, cv::Scalar(0, 0, 255), 3);
    }
    cv::resize(img, img, cv::Size(), 0.2, 0.2);
    cv::imshow("t", img);
    cv::waitKey(0);

    // // destroyModel();  // 删除后还是可以推理。。。

    std::cout << "..................." << std::endl;
    DET_RES* r2 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i){
        std::cout << r2[i].get_info() << std::endl;
    }
    std::cout << "..................." << std::endl;
    // TODO: 推理结果会有差异。。。。
    DET_RES* r3 = doInferenceBy3chImg(img.ptr(), img.rows, img.cols, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i){
        std::cout << r3[i].get_info() << std::endl;
    }
}