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

    std::string model_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-fp16.engine)"};
    initModel(model_file.c_str(), msg);
    std::vector<Item> image_files{
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr),
        // Item(R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)", nullptr)        
    };

    auto start = std::chrono::high_resolution_clock::now();
    std::for_each(std::execution::par, image_files.begin(), image_files.end(), [&](Item& item){
        std::cout << "----------" << std::endl;
        item.result = doInferenceByImgPth(item.file_path.c_str(), nullptr, 0.5f, item.det_num, msg);
    });
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cost = end - start;
    std::cout << "Inference total coast: " << cost.count() << "ms, Average cost: " << cost.count()/image_files.size() << "ms" << std::endl;

    Item item = image_files[0];
    std::cout << item.det_num << std::endl;
    for(int k{}; k<item.det_num; ++k){
        std::cout << item.result[k].get_info() << std::endl;
        // auto m = cv::Mat(cv::Size(i.result[k]->mask_w, i.result[k]->mask_h), i.result[k]->mask_type, i.result[k]->mask_data);
        // cv::imshow("mask_"+std::to_string(k), m);
        // cv::waitKey(0);
    }
}

void testInferenceSpeed(){
    // std::string model_file{R"(E:\Pretrained_models\YOLOv11\yolo11n-fp32.engine)"};
    std::string model_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-fp16.engine)"};
    std::vector<std::string> image_files{
        // R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)",  // 1560*1600  127ms
        // R"(E:\DataSets\imageNet\n02086240_Shih-Tzu.JPEG)",  // 1280*1024  49ms
        // R"(E:\le_trt\models\dog.jpg)",  // 500*374  26ms

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
    for(int i{}; i<100; i++){
        for(const auto& file:image_files){
            int det_num{};
            auto start = std::chrono::high_resolution_clock::now();
            auto r = doInferenceByImgPth(file.c_str(), nullptr, 0.5f, det_num, msg);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double, std::milli> spend = end - start;
            total_costs += spend.count();
            costs.push_back(spend.count());
            // for(int i{}; i<det_num; ++i){
            //     std::cout << "Result: " << r[i].get_info() << std::endl;
            //     auto m = cv::Mat(cv::Size(r[i].mask_w, r[i].mask_h), r[i].mask_type, r[i].mask_data);
            //     cv::imshow("t", m);
            //     cv::waitKey(0);
            // }
            std::cout << idx++ << " cost: " << spend.count() << "ms" << std::endl;
            freeResult(r, det_num);
        }
    }
    std::cout << "Average speed: " << total_costs/image_files.size() << "ms" << std::endl;    
    destroyModel();
}


int main(){
    std::cout << "TensorRT Test Demo" << std::endl;

    // testParallel();
    // return 0;

    testInferenceSpeed(); 
    return 0;

    printInfo();

    // run();

    std::string onnx_file{R"(E:\Pretrained_models\YOLOv11-seg\yolo11n-fp16.engine)"};
    // std::string img_file{R"(E:\le_pp3\Paddle\test\dataset\cat.jpg)"};
    // std::string img_file{R"(E:\le_trt\models\dog.jpg)"};
    std::string img_file{R"(D:\envs\ult_py311_cpu\Lib\site-packages\osam\_data\dogs.jpg)"};

    // initModel
    char msg[1024];
    initModel(onnx_file.c_str(), msg);
    std::cout << msg << std::endl;

    // doInferenceByMat
    cv::Mat img = cv::imread(img_file);
    int det_num{};
    auto start = std::chrono::high_resolution_clock::now();
    SEG_RES* res = doInferenceByImgPth(img_file.c_str(), nullptr, 0.5f, det_num, msg);
    // SEG_RES* res = doInferenceByImgMat(img, 0.5f, det_num, msg);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> cost = end-start;
    std::cout << "inference cost: " << cost.count() << "ms" << std::endl;
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