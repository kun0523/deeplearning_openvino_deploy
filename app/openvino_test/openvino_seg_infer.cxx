#include "inference.h"
#include <execution>

struct Item{
    std::string file_path{};
    int det_num{};
    SEG_RES* result = nullptr;

    Item(const std::string& img_pth, SEG_RES* res_ptr):file_path(img_pth), result(res_ptr){}
};

void testParallel(){
    char msg[1024];

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    initModel(onnx_file.c_str(), msg);
    std::vector<Item> image_files{
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr)        
    };

    auto start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par, image_files.begin(), image_files.end(), [&](Item& item){
        item.result = doInferenceByImgPth(item.file_path.c_str(), nullptr, 0.5f, item.det_num, msg);
    });
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cost = end - start;
    std::cout << "Inference total coast: " << cost.count() << "ms, Average cost: " << cost.count()/image_files.size() << "ms" << std::endl;
}

void testInferenceSpeed(){
    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
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

int main() {
    std::cout << "OpenVINO Segmentation Demo" << std::endl;
    
    // testParallel();
    // return 0;

    // speed test...
    testInferenceSpeed();
    return 0;

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-seg.onnx)"};
    // std::string img_file{R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)"};
    std::string img_file{R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"};

    // initModel
    char msg[1024];
    initModel(onnx_file.c_str(), msg);
    std::cout << msg << std::endl;

    // // doInferenceByMat
    cv::Mat img = cv::imread(img_file);
    int det_num{};
    // SEG_RES* res = doInferenceByImgMat(img, 0.5f, det_num, msg);
    // for(int i{}; i<det_num; ++i)
    //     std::cout << res[i].get_info() << std::endl;

    // doInferenceByImgPth()
    SEG_RES* res2 = doInferenceByImgPth(img_file.c_str(), nullptr, 0.3f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res2[i].get_info() << std::endl;
    return 0;

    // doInferenceBy3chImg()
    SEG_RES* res3 = doInferenceBy3chImg(img.data, img.rows, img.cols, 0.5f, det_num, msg);
    for(int i{}; i<det_num; ++i)
        std::cout << res3[i].get_info() << std::endl;



}