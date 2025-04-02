#include "inference.h"

void test_run(){

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    // std::string img_file{R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)"};
    std::string img_file{R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"};

    int det_num{};
    SEG_RES* result = run(img_file.c_str(), onnx_file.c_str(), det_num);

    std::cout << "Det Num: " << det_num << std::endl;
    for(int i{}; i<det_num; ++i){
        std::cout << "Det: " << i << " Result: " << result[i].cls << std::endl;
        std::cout << result[i].get_info() << std::endl;
        cv::Mat m{result[i].mask_h, result[i].mask_w, result[i].mask_type, result[i].mask_data};
        cv::imshow("ttt", m);
        cv::waitKey(0);
    }

    delete[] result;
}

void testInferenceSpeed(){
    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    std::vector<std::string> image_files{
        // R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",  // 1560*1600  187ms
        // R"(E:\DataSets\imageNet\n02086240_Shih-Tzu.JPEG)",  // 1280*1024  110ms
        // R"(E:\le_trt\models\dog.jpg)",  // 500*374  103ms

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
    initModel(onnx_file.c_str(), msg);
    for(const auto& file:image_files){
        int det_num{};
        auto start = std::chrono::high_resolution_clock::now();
        auto r = doInferenceByImgPth(file.c_str(), nullptr, 0.3f, det_num, msg);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> spend = end - start;
        total_costs += spend.count();
        costs.push_back(spend.count());
        std::cout << "Result: " << r->get_info() << std::endl;
        std::cout << "cost: " << spend.count() << "ms" << std::endl;
        // destroyModel(ptr);
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
}

int main(){
    printInfo();

    // speed test...
    testInferenceSpeed();
    return 0;

    // test_run();

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
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

    // doInferenceByImgPth()
    SEG_RES* res2 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.3f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res2[i].get_info() << std::endl;

    // doInferenceBy3chImg()
    SEG_RES* res3 = doInferenceBy3chImg(img.data, img.rows, img.cols, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res3[i].get_info() << std::endl;

}